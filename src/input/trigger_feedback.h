// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace Input::TriggerFeedback {
using Clock = std::chrono::steady_clock;
struct Sample {
    std::uint32_t packet{};
    float speed{};
    std::array<float, 4> omega{}, radius{};
    std::uint16_t flags{};
    std::uint8_t throttle{}, brake{};
};
template<class T> inline T Read(const std::uint8_t* data, unsigned offset) {
    T value;
    std::memcpy(&value, data + offset, sizeof(value));
    return value;
}
// GT Sport v1.69's little-endian 0x128-byte Simulator Interface packet.
inline bool Decode(const std::uint8_t* data, Sample& sample) {
    if (Read<std::uint32_t>(data, 0) != 0x47375330) return false;
    sample.packet = Read<std::uint32_t>(data, 0x70);
    sample.speed = Read<float>(data, 0x4c);
    sample.flags = Read<std::uint16_t>(data, 0x8e);
    sample.throttle = data[0x91];
    sample.brake = data[0x92];
    if (!std::isfinite(sample.speed) || sample.speed < 0 || sample.speed > 200) return false;
    for (unsigned i = 0; i < 4; ++i) {
        sample.omega[i] = Read<float>(data, 0xa4 + 4*i);
        sample.radius[i] = Read<float>(data, 0xb4 + 4*i);
        if (!std::isfinite(sample.omega[i]) || std::abs(sample.omega[i]) > 20000 ||
            !std::isfinite(sample.radius[i]) || sample.radius[i] < 0.05f || sample.radius[i] > 2)
            return false;
    }
    return true;
}
struct Feedback {
    std::uint8_t left{}, right{};
};
inline Feedback Calculate(const Sample& sample) {
    // Never actuate in menus, pause/loading, or at near-zero speed.
    if (!(sample.flags & 1) || (sample.flags & 6) || sample.speed < 3) return {};
    float lock = 0, spin = 0;
    for (unsigned i = 0; i < 4; ++i) {
        const float ratio = std::abs(sample.omega[i]) * sample.radius[i] / sample.speed;
        lock = std::max(lock, 1 - ratio);
        spin = std::max(spin, ratio - 1);
    }
    // This is wheel-lock estimation, not an ABS-active flag. The dead zones avoid
    // reacting to normal rolling-radius differences and small cornering differences.
    const float braking = std::clamp((lock - 0.08f) / 0.20f, 0.0f, 1.0f);
    const float traction = std::max(std::clamp((spin - 0.10f) / 0.30f, 0.0f, 1.0f),
                                    (sample.flags & 0x800) ? 0.65f : 0.0f);
    return {static_cast<std::uint8_t>(braking * sample.brake),
            static_cast<std::uint8_t>(traction * sample.throttle)};
}
inline std::mutex sample_mutex;
inline Sample latest_sample;
inline Clock::time_point updated{};
inline bool have_sample{};
inline std::atomic<bool> stopping{};
inline void Publish(const std::uint8_t* packet) {
    Sample sample;
    if (!Decode(packet, sample)) return;
    std::lock_guard lock(sample_mutex);
    if (!have_sample || sample.packet != latest_sample.packet) {
        latest_sample = sample;
        updated = Clock::now();
        have_sample = true;
    }
}
inline Feedback Current() {
    std::lock_guard lock(sample_mutex);
    if (stopping || !have_sample || Clock::now() - updated > std::chrono::milliseconds(250)) return {};
    return Calculate(latest_sample);
}
using Effect = std::array<std::uint8_t, 11>;
inline std::uint8_t Scale(std::uint8_t strength, unsigned percent) {
    return static_cast<std::uint8_t>(std::min(255u, (strength * std::min(percent, 200u) + 50u) / 100u));
}
inline Effect ScaleEffect(Effect effect, unsigned percent) {
    percent = std::min(percent, 200u);
    if (percent == 100 || effect[0] == 5) return effect;
    if (!percent) { effect = {}; effect[0] = 5; return effect; }
    auto amplitude = [percent](unsigned strength) {
        return std::min(8u, (strength * percent + 50u) / 100u);
    };
    if (effect[0] == 0x21 || effect[0] == 0x26) {
        unsigned zones = effect[1] | (unsigned{effect[2]} << 8);
        std::uint32_t packed = 0;
        for (unsigned i = 0; i < 4; ++i) packed |= std::uint32_t{effect[3+i]} << (8*i);
        std::uint32_t scaled = 0;
        for (unsigned i = 0; i < 10; ++i) if (zones & (1u << i)) {
            const unsigned strength = amplitude(((packed >> (3*i)) & 7u) + 1);
            if (!strength) zones &= ~(1u << i);
            else scaled |= (strength - 1) << (3*i);
        }
        if (!zones) { effect = {}; effect[0] = 5; return effect; }
        effect[1] = zones & 255;
        effect[2] = (zones >> 8) & 255;
        for (unsigned i = 0; i < 4; ++i) effect[3+i] = (scaled >> (8*i)) & 255;
    } else if (effect[0] == 0x25) {
        const unsigned strength = amplitude((effect[3] & 7) + 1);
        if (!strength) { effect = {}; effect[0] = 5; return effect; }
        effect[3] = strength - 1;
    }
    return effect;
}
inline Effect Vibration(std::uint8_t strength) {
    Effect effect{};
    effect[0] = 0x05; // Release resistance/vibration, including on pause and stale data.
    if (!strength) return effect;
    effect[0] = 0x26;
    effect[1] = 0xfe; // Zones 1-9; leave the start of the travel free.
    effect[2] = 0x03;
    const unsigned amplitude = (strength - 1) * 7 / 254;
    std::uint32_t packed = 0;
    for (unsigned zone = 1; zone < 10; ++zone) packed |= amplitude << (3*zone);
    for (unsigned i = 0; i < 4; ++i) effect[3+i] = (packed >> (8*i)) & 255;
    effect[9] = 25; // Native trigger vibration frequency, not a software rumble loop.
    return effect;
}
inline std::array<std::uint8_t, 47> DualSensePacket(const Effect& left, const Effect& right) {
    std::array<std::uint8_t, 47> packet{};
    packet[0] = 0x0c; // Enable ONLY right/left trigger effects; preserve LEDs/body rumble.
    std::copy(right.begin(), right.end(), packet.begin() + 10);
    std::copy(left.begin(), left.end(), packet.begin() + 21);
    return packet;
}
inline Effect PedalResistance(bool brake, unsigned percent) {
    // Increasing force through the travel, with the initial zone free. The brake
    // is firmer than the accelerator; percentages scale force, not zone positions.
    constexpr std::array<unsigned, 10> brake_force{0, 1, 1, 2, 2, 3, 4, 5, 6, 7};
    constexpr std::array<unsigned, 10> throttle_force{0, 1, 1, 1, 1, 2, 2, 2, 3, 3};
    Effect effect{};
    effect[0] = 0x21;
    unsigned zones = 0;
    std::uint32_t packed = 0;
    for (unsigned i = 0; i < 10; ++i) {
        const unsigned force = std::min(8u, ((brake ? brake_force[i] : throttle_force[i]) *
                                                  std::min(percent, 200u) + 50) / 100);
        if (!force) continue;
        zones |= 1u << i;
        packed |= (force - 1) << (3*i);
    }
    if (!zones) return Vibration(0);
    effect[1] = zones & 255;
    effect[2] = (zones >> 8) & 255;
    for (unsigned i = 0; i < 4; ++i) effect[3+i] = (packed >> (8*i)) & 255;
    return effect;
}
struct Output {
    Effect left = Vibration(0), right = Vibration(0);
    std::uint8_t left_rumble{}, right_rumble{};
};
inline Output CurrentOutput(unsigned percent) {
    std::lock_guard lock(sample_mutex);
    if (stopping || !have_sample || Clock::now() - updated > std::chrono::milliseconds(250) ||
        !(latest_sample.flags & 1) || (latest_sample.flags & 6)) return {};
    const auto feedback = Calculate(latest_sample);
    Output output;
    output.left_rumble = Scale(feedback.left, percent);
    output.right_rumble = Scale(feedback.right, percent);
    output.left = output.left_rumble ? Vibration(output.left_rumble) : PedalResistance(true, percent);
    output.right = output.right_rumble ? Vibration(output.right_rumble) : PedalResistance(false, percent);
    return output;
}
} // namespace Input::TriggerFeedback
