#include "input/trigger_feedback.h"
#include <iostream>
#include <stdexcept>

using namespace Input::TriggerFeedback;
static void Check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
int main() {
    Sample sample;
    sample.speed = 30;
    sample.flags = 1;
    sample.radius.fill(0.3f);
    sample.omega.fill(-100);
    sample.brake = sample.throttle = 255;
    Check(Calculate(sample).left == 0 && Calculate(sample).right == 0, "normal rolling is silent");
    sample.omega[0] = -50;
    Check(Calculate(sample).left == 255 && Calculate(sample).right == 0, "wheel lock affects brake only");
    sample.omega.fill(-100);
    sample.omega[2] = -150;
    Check(Calculate(sample).right == 255 && Calculate(sample).left == 0, "wheelspin affects accelerator only");
    sample.omega.fill(-100);
    sample.flags |= 0x800;
    Check(Calculate(sample).right > 0, "TCS activation affects accelerator");
    sample.throttle = 0;
    Check(Calculate(sample).right == 0, "released accelerator silent");
    for (unsigned flags : {0u, 3u, 5u}) {
        sample.flags = flags;
        Check(Calculate(sample).left == 0 && Calculate(sample).right == 0, "menu/pause/loading silent");
    }
    Check(Current().left == 0 && Current().right == 0, "no telemetry silent");
    sample.flags = 1;
    sample.speed = 0;
    Check(Calculate(sample).left == 0 && Calculate(sample).right == 0, "standstill silent");
    auto left = Vibration(120), right = Vibration(240);
    auto packet = DualSensePacket(left, right);
    Check(packet.size() == 47 && packet[0] == 0x0c && packet[1] == 0 && packet[38] == 0,
          "trigger-only SDL report preserves other effects");
    Check(std::equal(left.begin(), left.end(), packet.begin()+21) &&
          std::equal(right.begin(), right.end(), packet.begin()+10), "left/right report order");
    Check(Vibration(0)[0] == 5 && left[0] == 0x26 && left[9] == 25, "vibration and release encoding");
    Check(Scale(200, 0) == 0 && Scale(200, 50) == 100 && Scale(200, 100) == 200 &&
          Scale(200, 200) == 255, "strength gain and saturation");
    Check(ScaleEffect(left, 0)[0] == 5 && ScaleEffect(left, 100) == left,
          "explicit effect disabled or preserved");
    Check(ScaleEffect(Vibration(255), 200) == Vibration(255), "maximum effect saturates");
    Check(PedalResistance(true, 100)[0] == 0x21 && PedalResistance(false, 100)[0] == 0x21 &&
          PedalResistance(true, 0)[0] == 5, "progressive resistance and disable");
    Check(PedalResistance(true, 100) != PedalResistance(false, 100), "brake and accelerator have distinct curves");
    std::array<std::uint8_t, 0x128> raw{};
    Sample decoded;
    Check(!Decode(raw.data(), decoded), "invalid packet rejected");
    auto put = [&](unsigned offset, auto value) { std::memcpy(raw.data()+offset,&value,sizeof(value)); };
    put(0, std::uint32_t{0x47375330});
    put(0x4c, 30.0f);
    put(0x70, std::uint32_t{1});
    put(0x8e, std::uint16_t{1});
    raw[0x92] = 255;
    for (unsigned i=0;i<4;++i) { put(0xa4+4*i,-50.0f); put(0xb4+4*i,0.3f); }
    Check(Decode(raw.data(), decoded) && decoded.brake == 255, "documented packet offsets");
    Publish(raw.data());
    Check(Current().left == 255, "live data published");
    Check(CurrentOutput(100).left[0] == 0x26 && CurrentOutput(100).right[0] == 0x21,
          "braking vibration overrides left resistance only");
    Check(CurrentOutput(0).left[0] == 5 && CurrentOutput(0).right[0] == 5, "zero slider releases both pedals");
    { std::lock_guard lock(sample_mutex); updated = Clock::now()-std::chrono::seconds(1); }
    Publish(raw.data());
    Check(Current().left == 0, "repeated packet cannot revive stale effects");
    Check(CurrentOutput(100).left[0] == 5 && CurrentOutput(100).right[0] == 5, "stale resistance released");
    put(0x70, std::uint32_t{2});
    raw[0x92] = 0;
    Publish(raw.data());
    Check(CurrentOutput(100).left[0] == 0x21, "resistance restored after event");
    stopping = true;
    Check(CurrentOutput(100).left[0] == 5, "normal exit releases resistance");
    stopping = false;
    put(0xb4, 0.0f);
    Check(!Decode(raw.data(), decoded), "invalid radii rejected");
    std::cout << "PASS: telemetry decoding, lock/spin/TCS, gating, expiry, DualSense report\n";
}
