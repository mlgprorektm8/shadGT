// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Diagnostic bundles: the draw history and the one-pass capture (video_core/diag_bundle.h).

#include <cstdlib>
#include <deque>
#include <mutex>
#include <optional>

#include <boost/container/small_vector.hpp>
#include <nlohmann/json.hpp>

#include "common/io_file.h"
#include "common/logging/log.h"
#include "core/debug_state.h"
#include "core/memory.h"
#include "shader_recompiler/recompiler.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_rasterizer_diag.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {

namespace {
// Episode counting and delayed starts for armed bundles (see DiagBundle::Request).
u32 g_last_match_frame = ~0u;
u32 g_episodes = 0;
std::optional<VideoCore::DiagBundle::Request> g_delayed;
u32 g_delayed_start = 0;
} // namespace

namespace {
enum class DiagRole : u8 { Sampled, Target, Storage, Depth };

struct DiagImageRef {
    u64 uid;
    u32 slot;
    VAddr address;
    u32 width;
    u32 height;
    vk::Format format;
    u64 version;
    VideoCore::ImageFlagBits flags;
    DiagRole role;
};

struct DiagDrawRecord {
    u64 seq;
    u32 frame;
    bool compute;
    u32 count;
    std::array<u64, Shader::MaxStageTypes> stages{};
    boost::container::small_vector<DiagImageRef, 6> images;
    struct BufferRef {
        VAddr address;
        u32 size;
        bool written;
    };
    boost::container::small_vector<BufferRef, 4> buffers;
    std::string empty;
    /// While a bundle is captured: each stage's user data registers and the dwords loaded
    /// through them (the shader's flattened user data), as hex.
    std::string user_data;
};

std::mutex g_history_mutex;
std::deque<DiagDrawRecord> g_history;
u64 g_seq = 0;
constexpr size_t HistorySize = 150000;

constexpr std::array<const char*, Shader::MaxStageTypes> StageNames{"fs", "tcs", "tes",
                                                                    "vs", "gs",  "cs"};

const char* RoleName(DiagRole role) {
    switch (role) {
    case DiagRole::Sampled:
        return "read";
    case DiagRole::Target:
        return "target";
    case DiagRole::Storage:
        return "storage";
    case DiagRole::Depth:
        return "depth";
    }
    return "?";
}

nlohmann::json RecordJson(const DiagDrawRecord& record) {
    nlohmann::json shaders = nlohmann::json::object();
    for (size_t i = 0; i < record.stages.size(); ++i) {
        if (record.stages[i]) {
            shaders[StageNames[i]] = fmt::format("{:#x}", record.stages[i]);
        }
    }
    nlohmann::json images = nlohmann::json::array();
    for (const auto& ref : record.images) {
        images.push_back({{"uid", ref.uid},
                          {"slot", ref.slot},
                          {"address", fmt::format("{:#x}", ref.address)},
                          {"width", ref.width},
                          {"height", ref.height},
                          {"format", vk::to_string(ref.format)},
                          {"version", ref.version},
                          {"flags", static_cast<u32>(ref.flags)},
                          {"role", RoleName(ref.role)}});
    }
    nlohmann::json buffers = nlohmann::json::array();
    for (const auto& buffer : record.buffers) {
        buffers.push_back({{"address", fmt::format("{:#x}", buffer.address)},
                           {"size", buffer.size},
                           {"written", buffer.written}});
    }
    nlohmann::json line = {{"seq", record.seq},
                           {"frame", record.frame},
                           {"type", record.compute ? "dispatch" : "draw"},
                           {"count", record.count},
                           {"shaders", shaders},
                           {"images", images},
                           {"buffers", buffers}};
    if (!record.empty.empty()) {
        line["empty_bindings"] = record.empty;
    }
    if (!record.user_data.empty()) {
        line["user_data"] = record.user_data;
    }
    return line;
}

void WriteJson(const std::filesystem::path& path, const nlohmann::json& value) {
    std::ofstream{path, std::ios::trunc} << value.dump(2) << '\n';
}
} // namespace

void Rasterizer::RecordDiagHistory(const Pipeline* pipeline, bool compute, u32 count) {
    static std::once_flag crash_writer_once;
    std::call_once(crash_writer_once, [this] {
        VideoCore::DiagBundle::SetCrashWriter([this](const std::filesystem::path& dir) {
            DiagWriteHistory(dir / "draws.jsonl", true);
            WriteJson(dir / "manifest.json", {{"version", 1}, {"reason", "crash"}});
        });
    });

    DiagDrawRecord record{};
    record.compute = compute;
    record.count = count;
    record.frame = DebugState.GetFrameNum();
    for (const auto* info : pipeline->GetStages()) {
        if (info) {
            record.stages[static_cast<u32>(info->sw_stage)] = info->pgm_hash;
        }
    }
    const auto add = [&](VideoCore::ImageId id, DiagRole role) {
        if (!id) {
            return;
        }
        const auto& image = texture_cache.GetImage(id);
        for (const auto& existing : record.images) {
            if (existing.uid == image.image_uid && existing.role == role) {
                return;
            }
        }
        record.images.push_back({image.image_uid, id.index, image.info.guest_address,
                                 image.info.size.width, image.info.size.height,
                                 image.info.pixel_format, image.contents_version, image.flags,
                                 role});
    };
    if (!compute) {
        const auto& key = static_cast<const GraphicsPipeline*>(pipeline)->GetGraphicsKey();
        for (u32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
            add(cb_descs[cb].image_id, DiagRole::Target);
        }
        add(db_desc.first, DiagRole::Depth);
    }
    for (const auto id : bound_images) {
        const auto& image = texture_cache.GetImage(id);
        if (image.binding.is_target) {
            continue;
        }
        const bool storage =
            image.binding.force_general ||
            std::ranges::find(diag_storage_images, id) != diag_storage_images.end();
        add(id, storage ? DiagRole::Storage : DiagRole::Sampled);
    }
    for (const auto& bound : bound_buffers) {
        if (bound.guest_address != 0) {
            record.buffers.push_back({bound.guest_address, bound.size, bound.is_written});
        }
    }
    record.empty = diag_empty_bindings;
    if (diag_capture) {
        for (const auto* info : pipeline->GetStages()) {
            if (!info) {
                continue;
            }
            record.user_data += fmt::format("{}{}:", record.user_data.empty() ? "" : " ",
                                            StageNames[static_cast<u32>(info->sw_stage)]);
            for (const u32 dword : info->user_data) {
                record.user_data += fmt::format(" {:08x}", dword);
            }
            record.user_data += " |";
            for (const u32 dword : info->flattened_ud_buf) {
                record.user_data += fmt::format(" {:08x}", dword);
            }
        }
    }

    // A bundle armed on a target size or a shader starts at the draw that matches it.
    if (!diag_capture) {
        if (const auto armed = VideoCore::DiagBundle::Armed()) {
            using VideoCore::DiagBundle::Trigger;
            bool fire = false;
            if (armed->trigger == Trigger::Shader) {
                fire = std::ranges::find(record.stages, armed->shader) != record.stages.end();
            } else if (armed->trigger == Trigger::Target && !compute) {
                fire = std::ranges::any_of(record.images, [&](const DiagImageRef& ref) {
                    return ref.role == DiagRole::Target && ref.width == armed->width &&
                           ref.height == armed->height;
                });
            }
            if (fire && (armed->skip || armed->repeat > 1)) {
                // Count matching episodes: draws to the trigger within 120 frames of the last
                // match belong to the same one (one thumbnail render spans several frames).
                const u32 frame = DebugState.GetFrameNum();
                if (g_last_match_frame == ~0u || frame - g_last_match_frame > 120) {
                    ++g_episodes;
                    LOG_WARNING(Render_Vulkan, "DIAG-BUNDLE: '{}' matched at frame {} (episode {})",
                                armed->reason, frame, g_episodes);
                }
                g_last_match_frame = frame;
                fire = g_episodes > armed->skip;
            }
            if (fire) {
                VideoCore::DiagBundle::Disarm();
                if (armed->delay) {
                    g_delayed = *armed;
                    g_delayed_start = DebugState.GetFrameNum() + armed->delay;
                    LOG_WARNING(Render_Vulkan, "DIAG-BUNDLE: '{}' starts at frame {}",
                                armed->reason, g_delayed_start);
                } else {
                    DiagStart(*armed);
                }
            }
        }
    }

    {
        std::scoped_lock lk{g_history_mutex};
        record.seq = ++g_seq;
        if (diag_capture) {
            for (const auto& ref : record.images) {
                diag_capture->touched.try_emplace(ref.uid, ref.slot);
            }
            for (const u64 hash : record.stages) {
                if (hash) {
                    diag_capture->shaders.insert(hash);
                }
            }
            for (const auto& [address, value] : diag_sharp_reads) {
                if (diag_capture->sharp_reads.size() < 20000) {
                    diag_capture->sharp_reads.try_emplace(address, value, record.seq);
                }
            }
            for (const auto& buffer : record.buffers) {
                if (diag_capture->buffers.size() < 20000) {
                    diag_capture->buffers[{buffer.address, buffer.size}] |= buffer.written;
                }
            }
        }
        g_history.push_back(std::move(record));
        if (g_history.size() > HistorySize) {
            g_history.pop_front();
        }
    }
}

void Rasterizer::DiagWriteHistory(const std::filesystem::path& path, bool try_lock) {
    std::unique_lock lk{g_history_mutex, std::defer_lock};
    if (try_lock ? !lk.try_lock() : (lk.lock(), false)) {
        return;
    }
    std::ofstream out{path, std::ios::trunc};
    for (const auto& record : g_history) {
        out << RecordJson(record).dump() << '\n';
    }
}

void Rasterizer::DiagBeforeWork() {
    if (!diag_capture && g_delayed && DebugState.GetFrameNum() >= g_delayed_start) {
        const auto request = *g_delayed;
        g_delayed.reset();
        DiagStart(request);
    }
    if (!diag_capture) {
        if (const auto armed = VideoCore::DiagBundle::Armed();
            armed && armed->trigger == VideoCore::DiagBundle::Trigger::Now) {
            VideoCore::DiagBundle::Disarm();
            DiagStart(*armed);
        }
        return;
    }
    const u32 frame = DebugState.GetFrameNum();
    if (frame == diag_capture->frame) {
        return;
    }
    diag_capture->frame = frame;
    DiagSnapshot();
    if (++diag_capture->frames_done >= diag_capture->request.frames) {
        DiagFinish();
    }
}

void Rasterizer::DiagStart(const VideoCore::DiagBundle::Request& request) {
    auto capture = std::make_unique<DiagCapture>();
    capture->request = request;
    capture->dir = VideoCore::DiagBundle::NewBundleDir(request.reason);
    capture->frame = DebugState.GetFrameNum();
    {
        std::scoped_lock lk{g_history_mutex};
        capture->start_seq = g_seq + 1;
    }
    std::error_code ec;
    std::filesystem::create_directories(capture->dir / "images", ec);
    capture->images.open(capture->dir / "images.jsonl", std::ios::trunc);
    VideoCore::DiagBundle::BeginPm4Capture(capture->dir / "pm4");
    LOG_WARNING(Render_Vulkan, "DIAG-BUNDLE: capturing '{}' from draw {} into {}", request.reason,
                capture->start_seq, capture->dir.string());
    diag_capture = std::move(capture);
}

void Rasterizer::DiagSnapshot() {
    auto& capture = *diag_capture;
    const u32 snapshot = ++capture.snapshots;
    u64 after_seq;
    std::unordered_map<u64, u32> touched;
    {
        std::scoped_lock lk{g_history_mutex};
        after_seq = g_seq;
        touched.swap(capture.touched);
    }
    static const u64 MaxBundleImageBytes = [] {
        const char* env = std::getenv("SHADGT_BUNDLE_MAX_MB");
        return (env && *env ? std::strtoull(env, nullptr, 10) : 6144ULL) << 20;
    }();
    for (const auto& [uid, slot] : touched) {
        const VideoCore::ImageId id{slot};
        nlohmann::json line = {{"snapshot", snapshot},
                               {"after_seq", after_seq},
                               {"frame", capture.frame},
                               {"uid", uid}};
        if (!texture_cache.IsImageAlive(id, uid)) {
            line["state"] = "freed";
            capture.images << line.dump() << '\n';
            continue;
        }
        const auto& image = texture_cache.GetImage(id);
        const auto& info = image.info;
        const u64 texels = u64(info.size.width) * info.size.height;
        line.update({{"slot", slot},
                     {"address", fmt::format("{:#x}", info.guest_address)},
                     {"width", info.size.width},
                     {"height", info.size.height},
                     {"format", vk::to_string(info.pixel_format)},
                     {"depth", info.props.is_depth},
                     {"tiled", info.props.is_tiled},
                     {"version", image.contents_version},
                     {"flags", static_cast<u32>(image.flags)}});
        nlohmann::json mods = nlohmann::json::array();
        for (const auto& mod : VideoCore::RecentImageModifications(uid, 8)) {
            std::string_view file = mod.file ? mod.file : "?";
            if (const auto slash = file.find_last_of("/\\"); slash != std::string_view::npos) {
                file.remove_prefix(slash + 1);
            }
            mods.push_back(fmt::format("ver {} by {}:{}", mod.version, file, mod.line));
        }
        line["modifications"] = mods;
        const u64 bytes = texels * std::max(info.num_bits / 8, 1u);
        if (const auto it = capture.dumped.find(uid);
            it != capture.dumped.end() && it->second.first == image.contents_version) {
            // Unchanged since its last dump (textures, untouched targets).
            line["file"] = it->second.second;
            line["unchanged"] = true;
        } else if (info.props.is_block || texels > 4096ULL * 4096 ||
                   capture.image_bytes + bytes > MaxBundleImageBytes) {
            line["state"] = "not dumped";
            ++capture.images_skipped;
        } else {
            const auto name = fmt::format("s{}_uid{}_{}x{}_{}.bin", snapshot, uid, info.size.width,
                                          info.size.height, vk::to_string(info.pixel_format));
            if (texture_cache.DumpImage(id, capture.dir / "images" / name)) {
                line["file"] = "images/" + name;
                capture.dumped[uid] = {image.contents_version, "images/" + name};
                capture.image_bytes += bytes;
                ++capture.images_written;
            } else {
                line["state"] = "not dumped";
                ++capture.images_skipped;
            }
        }
        capture.images << line.dump() << '\n';
    }
    capture.images.flush();
    LOG_WARNING(Render_Vulkan, "DIAG-BUNDLE: snapshot {} after draw {}: {} images", snapshot,
                after_seq, touched.size());
}

void Rasterizer::DiagFinish() {
    auto& capture = *diag_capture;
    VideoCore::DiagBundle::EndPm4Capture();
    u64 end_seq;
    {
        std::scoped_lock lk{g_history_mutex};
        end_seq = g_seq;
    }
    DiagWriteHistory(capture.dir / "draws.jsonl", false);

    // Shaders: guest code, a readable listing and the SPIR-V of each permutation.
    std::error_code ec;
    std::filesystem::create_directories(capture.dir / "shaders", ec);
    nlohmann::json shaders = nlohmann::json::array();
    for (const u64 hash : capture.shaders) {
        const auto* program = pipeline_cache.FindProgram(hash);
        nlohmann::json entry = {{"hash", fmt::format("{:#x}", hash)}};
        if (!program) {
            entry["state"] = "not found";
        } else {
            const auto base = capture.dir / "shaders" / fmt::format("{:#x}", hash);
            if (!program->guest_code.empty()) {
                auto bin = base;
                bin += ".gcn.bin";
                Common::FS::IOFile{bin, Common::FS::FileAccessMode::Create}.WriteSpan(
                    std::span{program->guest_code});
                auto txt = base;
                txt += ".gcn.txt";
                std::ofstream{txt} << Shader::ListGcnCode(program->guest_code);
                entry["gcn"] = fmt::format("shaders/{:#x}.gcn.txt", hash);
            }
            nlohmann::json perms = nlohmann::json::array();
            for (size_t i = 0; i < program->modules.size(); ++i) {
                const auto& module = program->modules[i];
                if (!module.info) {
                    continue;
                }
                nlohmann::json perm = {{"index", i}, {"translated", bool(module.module)}};
                if (!module.spv.empty()) {
                    const auto name = fmt::format("{:#x}_p{}.spv", hash, i);
                    Common::FS::IOFile{capture.dir / "shaders" / name,
                                       Common::FS::FileAccessMode::Create}
                        .WriteSpan(std::span{module.spv});
                    perm["spv"] = "shaders/" + name;
                }
                perms.push_back(perm);
            }
            entry["permutations"] = perms;
        }
        shaders.push_back(entry);
    }
    WriteJson(capture.dir / "shaders.json", shaders);

    // Buffers: the guest bytes of the ranges bound during the capture (up to 1 MB each, 256 MB
    // in all). Ranges a draw or dispatch wrote may be newer on the GPU than in guest memory.
    std::filesystem::create_directories(capture.dir / "buffers", ec);
    nlohmann::json buffers = nlohmann::json::array();
    u64 buffer_bytes = 0;
    for (const auto& [range, written] : capture.buffers) {
        const auto [address, size] = range;
        nlohmann::json entry = {{"address", fmt::format("{:#x}", address)},
                                {"size", size},
                                {"written_by_gpu", written}};
        if (size <= 1_MB && buffer_bytes + size <= 256_MB &&
            memory->IsValidMapping(address, size)) {
            const auto name = fmt::format("{:#x}_{:#x}.bin", address, size);
            Common::FS::IOFile{capture.dir / "buffers" / name, Common::FS::FileAccessMode::Create}
                .WriteRaw<u8>(std::bit_cast<const u8*>(address), size);
            entry["file"] = "buffers/" + name;
            buffer_bytes += size;
        }
        buffers.push_back(entry);
    }
    WriteJson(capture.dir / "buffers.json", buffers);

    // Descriptors: the V# dwords of empty bindings as the draws read them, and the same guest
    // memory now. A dword that changed was written after the draw read it (a descriptor read too
    // early, or memory reused).
    nlohmann::json descriptors = nlohmann::json::array();
    u32 changed = 0;
    for (const auto& [address, read] : capture.sharp_reads) {
        const auto [value, seq] = read;
        nlohmann::json entry = {{"address", fmt::format("{:#x}", address)},
                                {"read", fmt::format("{:#010x}", value)},
                                {"first_draw", seq}};
        if (memory->IsValidMapping(address, 4)) {
            const u32 now = *std::bit_cast<const u32*>(address);
            entry["now"] = fmt::format("{:#010x}", now);
            entry["changed"] = now != value;
            changed += now != value ? 1 : 0;
        }
        descriptors.push_back(entry);
    }
    WriteJson(capture.dir / "descriptors.json", descriptors);
    LOG_WARNING(Render_Vulkan,
                "DIAG-BUNDLE: {} of {} empty-binding descriptor dwords changed "
                "after the draws read them",
                changed, capture.sharp_reads.size());

    const auto& request = capture.request;
    WriteJson(capture.dir / "manifest.json",
              {{"version", 1},
               {"reason", request.reason},
               {"trigger", static_cast<u32>(request.trigger)},
               {"trigger_size", fmt::format("{}x{}", request.width, request.height)},
               {"trigger_shader", fmt::format("{:#x}", request.shader)},
               {"frames", request.frames},
               {"start_seq", capture.start_seq},
               {"end_seq", end_seq},
               {"snapshots", capture.snapshots},
               {"images_dumped", capture.images_written},
               {"images_not_dumped", capture.images_skipped},
               {"image_bytes", capture.image_bytes},
               {"shaders", capture.shaders.size()},
               {"buffers", capture.buffers.size()},
               {"diag_mode", VideoCore::DiagBundle::Enabled()}});
    LOG_WARNING(Render_Vulkan, "DIAG-BUNDLE: written to {} (draws {}..{}, {} images, {} shaders)",
                capture.dir.string(), capture.start_seq, end_seq, capture.images_written,
                capture.shaders.size());
    // Repeat: arm the same trigger for the next episode.
    if (auto next = capture.request; next.repeat > 1) {
        --next.repeat;
        next.skip = g_episodes;
        VideoCore::DiagBundle::Arm(std::move(next));
    }
    diag_capture.reset();
}

} // namespace Vulkan
