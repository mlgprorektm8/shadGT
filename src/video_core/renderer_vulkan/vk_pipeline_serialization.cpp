// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_stdinc.h>
#include <SDL3/SDL_video.h>

#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Serialization {
/* You should increment versions below once corresponding serialization scheme is changed. */
// Synthetic dual-source exports change the fragment shader interface. Version 13 adds the
// general swizzled factor blend, which also changes the runtime color-buffer layout.
static constexpr u32 ShaderBinaryVersion = 15u;
// Stored output metadata and runtime color-buffer flags include swizzled-alpha emulation.
// Version 13: stored SRT walker code also records flattened source addresses.
// Version 14: stored specializations no longer include compile-time runtime-info changes.
static constexpr u32 ShaderMetaVersion = 14u;
static constexpr u32 PipelineKeyVersion = 8u;
} // namespace Serialization

namespace Vulkan {

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{1}); // compute

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("c_{:#018x}", key.value), ar.TakeOff());
}

void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar{};
    Serialization::Writer pldata{ar};

    pldata.Write(Serialization::PipelineKeyVersion);
    pldata.Write(u32{0}); // graphics

    key.Serialize(ar);
    sdata.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::PipelineKey,
                                       fmt::format("g_{:#018x}", hash), ar.TakeOff());
}

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Serialization::Archive ar;
    Serialization::Writer meta{ar};

    meta.Write(Serialization::ShaderMetaVersion);
    meta.Write(Serialization::ShaderBinaryVersion);

    meta.Write(perm_hash);
    meta.Write(perm_idx);

    spec.Serialize(ar);
    info.Serialize(ar);

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", perm_hash), ar.TakeOff());
}

void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Storage::DataBase::Instance().IsOpened()) {
        return;
    }

    Storage::DataBase::Instance().Save(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx),
                                       std::move(spv));
}

bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    Shader::StageSpecialization& spec, size_t& perm_idx) {
    Serialization::Reader meta{ar};

    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion) {
        return false;
    }

    u32 binary_version{};
    meta.Read(binary_version);
    if (binary_version != Serialization::ShaderBinaryVersion) {
        return false;
    }

    u64 perm_hash_ar{};
    meta.Read(perm_hash_ar);
    meta.Read(perm_idx);

    spec.Deserialize(ar);
    info.Deserialize(ar, perm_hash_ar);
    return true;
}

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};
    key.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};
    key.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    // Nothing here yet
    return;
}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    // Nothing here yet
    return true;
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    compute_key.Deserialize(ar);

    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    std::vector<u8> meta_blob;
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                       fmt::format("{:#018x}", compute_key.value), meta_blob);
    if (meta_blob.empty()) {
        return false;
    }

    Serialization::Archive meta_ar{std::move(meta_blob)};

    if (!LoadPipelineStage(meta_ar, 0)) {
        return false;
    }

    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);

    it.value() =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *infos[0], modules[0], sdata, true);

    infos.fill(nullptr);
    modules.fill(nullptr);

    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer key{ar};

    key.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader key{ar};

    key.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer sdata{ar};

    sdata.Write(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Write(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Write(&divisors, sizeof(divisors));
    sdata.Write(multisampling);
    sdata.Write(tcs);
    sdata.Write(tes);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader sdata{ar};

    sdata.Read(&vertex_attributes, sizeof(vertex_attributes));
    sdata.Read(&vertex_bindings, sizeof(vertex_bindings));
    sdata.Read(&divisors, sizeof(divisors));
    sdata.Read(multisampling);
    sdata.Read(tcs);
    sdata.Read(tes);
    return true;
}

// Worker threads that build preloaded graphics pipelines (see WarmUp).
struct PipelineCache::PreloadQueue {
    using Task = std::function<std::unique_ptr<GraphicsPipeline>()>;
    using Result = std::pair<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>>;

    explicit PreloadQueue(u32 num_workers) {
        for (u32 i = 0; i < num_workers; ++i) {
            workers.emplace_back([this] { Work(); });
        }
    }

    ~PreloadQueue() {
        Finish();
    }

    void Push(const GraphicsPipelineKey& key, Task&& task) {
        {
            std::scoped_lock lk{mutex};
            tasks.emplace_back(key, std::move(task));
        }
        ++num_queued;
        cv.notify_one();
    }

    std::vector<Result> Finish() {
        {
            std::scoped_lock lk{mutex};
            closing = true;
        }
        cv.notify_all();
        workers.clear();
        return std::move(results);
    }

    u32 NumQueued() const {
        return num_queued.load();
    }

    u32 NumDone() const {
        return num_done.load();
    }

private:
    void Work() {
        while (true) {
            std::pair<GraphicsPipelineKey, Task> item;
            {
                std::unique_lock lk{mutex};
                cv.wait(lk, [&] { return closing || !tasks.empty(); });
                if (tasks.empty()) {
                    return;
                }
                item = std::move(tasks.front());
                tasks.pop_front();
            }
            auto pipeline = item.second();
            {
                std::scoped_lock lk{mutex};
                results.emplace_back(item.first, std::move(pipeline));
            }
            ++num_done;
        }
    }

    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::pair<GraphicsPipelineKey, Task>> tasks;
    std::vector<Result> results;
    bool closing{};
    std::atomic<u32> num_queued{};
    std::atomic<u32> num_done{};
    std::vector<std::jthread> workers;
};

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    graphics_key.Deserialize(ar);

    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    for (int stage_idx = 0; stage_idx < MaxShaderStages; ++stage_idx) {
        const auto& hash = graphics_key.stage_hashes[stage_idx];
        if (!hash) {
            continue;
        }

        std::vector<u8> meta_blob;
        Storage::DataBase::Instance().Load(Storage::BlobType::ShaderMeta,
                                           fmt::format("{:#018x}", hash), meta_blob);
        if (meta_blob.empty()) {
            return false;
        }

        Serialization::Archive meta_ar{std::move(meta_blob)};

        if (!LoadPipelineStage(meta_ar, stage_idx)) {
            return false;
        }
    }

    const auto [it, is_new] = graphics_pipelines.try_emplace(graphics_key);
    ASSERT(is_new);

    if (preload_queue) {
        // Driver compilation is the slow part of the precompile, so it runs on worker threads.
        // Copy everything this load filled in; the members are reused by the next load.
        preload_queue->Push(
            graphics_key,
            [this, key = graphics_key, stage_infos = infos, stage_runtime = runtime_infos,
             fetch = fetch_shader ? std::optional{*fetch_shader} : std::nullopt,
             stage_modules = modules, sdata]() mutable {
                return std::make_unique<GraphicsPipeline>(
                    instance, scheduler, desc_heap, profile, key, *pipeline_cache, stage_infos,
                    stage_runtime, fetch ? &*fetch : nullptr, stage_modules, sdata, true);
            });
    } else {
        it.value() = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, graphics_key, *pipeline_cache, infos,
            runtime_infos, fetch_shader, modules, sdata, true);
    }

    infos.fill(nullptr);
    modules.fill(nullptr);
    fetch_shader = nullptr;

    return true;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t stage) {
    auto info = std::make_unique<Shader::Info>();
    Shader::StageSpecialization spec{};
    spec.info = info.get();
    size_t perm_idx{};
    if (!LoadShaderMeta(ar, *info, spec, perm_idx)) {
        return false;
    }

    std::vector<u32> spv{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", info->pgm_hash, perm_idx), spv);
    if (spv.empty()) {
        return false;
    }

    // Permutation hash depends on shader variation index. To prevent collisions, we need insert it
    // at the exact position rather than append

    vk::ShaderModule module{};

    auto [it_pgm, new_program] = program_cache.try_emplace(info->pgm_hash);
    if (new_program) {
        module = CompileSPV(spv, instance.GetDevice());
        it_pgm.value() = std::make_unique<Program>();
    } else {
        const auto& it = std::ranges::find(it_pgm.value()->modules, spec, &Program::Module::spec);
        if (it != it_pgm.value()->modules.end()) {
            // A matching permutation is valid only at its original index. A different index means
            // the store holds entries from more than one cache generation, so this pipeline is
            // left to compile at runtime.
            const auto idx = std::distance(it_pgm.value()->modules.begin(), it);
            if (perm_idx != idx) {
                LOG_WARNING(Render_Vulkan,
                            "Cached permutation {} of {}_{:x} conflicts with index {}, skipping "
                            "preload",
                            perm_idx, info->hw_stage, info->pgm_hash, idx);
                return false;
            }
            module = it->module;
            infos[stage] = it->info.get();
            modules[stage] = module;
            if (auto& fetch = it->spec.fetch_shader_data; !fetch.Empty()) {
                fetch_shader = &fetch;
            }
            return true;
        } else {
            if (perm_idx < it_pgm.value()->modules.size() &&
                it_pgm.value()->modules[perm_idx].info) {
                LOG_WARNING(Render_Vulkan,
                            "Conflicting metadata for cached permutation {} of {}_{:x}, "
                            "skipping preload",
                            perm_idx, info->hw_stage, info->pgm_hash);
                return false;
            }
            module = CompileSPV(spv, instance.GetDevice());
        }
    }
    it_pgm.value()->InsertPermut(module, std::move(spec), std::move(info), perm_idx);

    infos[stage] = it_pgm.value()->modules[perm_idx].info.get();
    modules[stage] = module;
    if (auto& fetch = it_pgm.value()->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        fetch_shader = &fetch;
    }

    return true;
}

void PipelineCache::WarmUp() {
    if (!EmulatorSettings.IsPipelineCacheEnabled()) {
        return;
    }

    Storage::DataBase::Instance().Open();

    // Check if cache is compatible
    std::vector<u8> profile_data{};
    Storage::DataBase::Instance().Load(Storage::BlobType::ShaderProfile, "profile", profile_data);
    if (profile_data.empty()) {
        Storage::DataBase::Instance().FinishPreload();

        profile_data.resize(sizeof(profile));
        std::memcpy(profile_data.data(), &profile, sizeof(profile));
        Storage::DataBase::Instance().Save(Storage::BlobType::ShaderProfile, "profile",
                                           std::move(profile_data));
        return;
    }
    if (profile_data.size() != sizeof(Shader::Profile)) {
        LOG_WARNING(Render,
                    "Pipeline cache profile has unexpected size ({} != {}). Ignoring the cache",
                    profile_data.size(), sizeof(Shader::Profile));
        Storage::DataBase::Instance().Close();
        return;
    }

    Shader::Profile cached_profile{};
    std::memcpy(&cached_profile, profile_data.data(), sizeof(cached_profile));
    if (cached_profile != profile) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current system. Ignoring the cache");
        Storage::DataBase::Instance().Close();
        return;
    }

    u32 num_pipelines{};
    u32 num_total_pipelines{};

    // Precompile every cached pipeline before the game starts. This runs on the thread that
    // owns the window, so show progress in the title and keep pumping window events; otherwise
    // the OS reports the window as not responding until the precompile ends.
    u32 num_cached{};
    Storage::DataBase::Instance().ForEachBlob(Storage::BlobType::PipelineKey,
                                              [&](std::vector<u8>&&) { ++num_cached; });
    int num_windows{};
    SDL_Window** windows = SDL_GetWindows(&num_windows);
    SDL_Window* window = windows && num_windows > 0 ? windows[0] : nullptr;
    SDL_free(windows);
    const std::string window_title = window ? SDL_GetWindowTitle(window) : "";
    auto last_progress = std::chrono::steady_clock::now();
    const auto report_progress = [&](u32 done, bool force) {
        const auto now = std::chrono::steady_clock::now();
        if (!window || (!force && now - last_progress < std::chrono::milliseconds(100))) {
            return;
        }
        last_progress = now;
        const auto title =
            fmt::format("{} - Compiling shaders {} / {} ({}%)", window_title, done, num_cached,
                        num_cached ? u64(done) * 100 / num_cached : 100);
        SDL_SetWindowTitle(window, title.c_str());
        SDL_PumpEvents();
    };

    const u32 num_workers = std::clamp(std::thread::hardware_concurrency(), 2u, 14u) - 1;
    std::optional<PreloadQueue> queue;
    if (num_cached > 0) {
        LOG_INFO(Render, "Precompiling {} cached pipelines on {} threads", num_cached,
                 num_workers);
        queue.emplace(num_workers);
        preload_queue = &*queue;
        report_progress(0, true);
    }
    u32 num_direct{}; // compute pipelines and rejected entries finish on this thread

    Storage::DataBase::Instance().ForEachBlob(
        Storage::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
            ++num_total_pipelines;
            report_progress(num_direct + queue->NumDone(), false);

            Serialization::Archive ar{std::move(data)};
            Serialization::Reader pldata{ar};

            u32 version{};
            pldata.Read(version);
            if (version != Serialization::PipelineKeyVersion) {
                ++num_direct;
                return;
            }

            u32 is_compute{};
            pldata.Read(is_compute);

            bool result{};
            const u32 queued_before = queue->NumQueued();
            if (is_compute) {
                result = LoadComputePipeline(ar);
            } else {
                result = LoadGraphicsPipeline(ar);
            }
            if (queue->NumQueued() == queued_before) {
                ++num_direct;
            }

            if (result) {
                ++num_pipelines;
            }
        });

    if (queue) {
        // Wait for the workers, keeping the window responsive, and save the driver cache now
        // and then so an interrupted precompile is not lost.
        u32 saved_at{};
        while (queue->NumDone() < queue->NumQueued()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            const u32 done = queue->NumDone();
            report_progress(num_direct + done, false);
            if (done - saved_at >= 1000) {
                saved_at = done;
                SaveDriverCache(false);
            }
        }
        for (auto& [key, pipeline] : queue->Finish()) {
            graphics_pipelines[key] = std::move(pipeline);
        }
        preload_queue = nullptr;
    }

    if (window) {
        SDL_SetWindowTitle(window, window_title.c_str());
    }    LOG_INFO(Render, "Preloaded {} pipelines", num_pipelines);
    if (num_total_pipelines > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total_pipelines - num_pipelines);
    }

    Storage::DataBase::Instance().FinishPreload();
}

void PipelineCache::Sync() {
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar, u64 walker_key) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);

    return srt_info.Deserialize(ar, walker_key);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer fetch{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));

    fetch.Write(size);
    fetch.Write(vertex_offset_sgpr);
    fetch.Write(instance_offset_sgpr);
    fetch.Write(attributes);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader fetch{ar};

    fetch.Read(size);
    fetch.Read(vertex_offset_sgpr);
    fetch.Read(instance_offset_sgpr);
    fetch.Read(attributes);

    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer srt{ar};

    srt.Write(this, sizeof(*this));
    if (walker_func_size) {
        srt.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar, u64 walker_key) {
    Serialization::Reader srt{ar};

    srt.Read(this, sizeof(*this));

    if (walker_func_size) {
        walker_func = RegisterWalkerCode(ar.CurrPtr(), walker_func_size, walker_key);
        ar.Advance(walker_func_size);
    }

    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer spec{ar};

    spec.Write(start);
    spec.Write(runtime_info);

    spec.Write(bitset.to_string());

    if (!fetch_shader_data.Empty()) {
        spec.Write(sizeof(fetch_shader_data));
        fetch_shader_data.Serialize(ar);
    } else {
        spec.Write(size_t{0});
    }

    spec.Write(vs_attribs);
    spec.Write(buffers);
    spec.Write(images);
    spec.Write(fmasks);
    spec.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader spec{ar};

    spec.Read(start);
    spec.Read(runtime_info);

    std::string bits{};
    spec.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);

    u64 fetch_data_size{};
    spec.Read(fetch_data_size);

    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }

    spec.Read(vs_attribs);
    spec.Read(buffers);
    spec.Read(images);
    spec.Read(fmasks);
    spec.Read(samplers);

    return true;
}

} // namespace Shader
