// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <semaphore>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/perf_monitor.h"
#include "common/thread.h"
#include "shader_recompiler/backend/spirv/emit_spirv_discard_frag.h"
#include "shader_recompiler/backend/spirv/emit_spirv_quad_rect.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

namespace Vulkan {

#ifdef _WIN32
// DIAG-100: every thread's stack to user\log\<file_name> (tools\stackdump.exe next to
// GTSport.exe), for a build or the GPU stuck for long.
void DumpThreadStacks(const wchar_t* file_name) {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    const std::filesystem::path dir = std::filesystem::path{exe}.parent_path();
    const auto tool = dir / "tools" / "stackdump.exe";
    if (!std::filesystem::exists(tool)) {
        return;
    }
    const auto out = dir / "user" / "log" / file_name;
    std::wstring command =
        L"\"" + tool.wstring() + L"\" " + std::to_wstring(GetCurrentProcessId()) + L" 512";
    SECURITY_ATTRIBUTES sa{sizeof(sa), nullptr, TRUE};
    HANDLE file = CreateFileW(out.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &sa, CREATE_ALWAYS,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return;
    }
    STARTUPINFOW si{sizeof(si)};
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = file;
    si.hStdError = file;
    PROCESS_INFORMATION pi{};
    if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr,
                       nullptr, &si, &pi)) {
        WaitForSingleObject(pi.hProcess, 30000);
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
        LOG_WARNING(Render_Vulkan, "DIAG-100: thread stacks written to {}", out.string());
    }
    CloseHandle(file);
}
#else
void DumpThreadStacks(const wchar_t*) {}
#endif

// FIX-104: pipeline creations in progress (frame generation pauses meanwhile).
static std::atomic<u32> g_builds_running{0};
u32 PipelineBuildsRunning() {
    return g_builds_running.load(std::memory_order_relaxed);
}

namespace {

// DIAG-049: graphics pipeline creations the driver takes long on, logged every 2 s while they
// run with the memory the process has committed. On GT Sport's race summary the driver's
// compiler used up all memory ("LLVM ERROR: out of memory").
class BuildWatch {
public:
    BuildWatch(std::string name_, const char* what_) : name{std::move(name_)}, what{what_} {
        g_builds_running.fetch_add(1, std::memory_order_relaxed); // FIX-104
        auto& state = State();
        std::scoped_lock lk{state.mutex};
        id = ++state.next_id;
        state.running.emplace(id, Entry{name, what, std::chrono::steady_clock::now(), false, 0});
        if (!state.watcher_started) {
            state.watcher_started = true;
            std::thread{Watch}.detach();
        }
    }
    ~BuildWatch() {
        g_builds_running.fetch_sub(1, std::memory_order_relaxed); // FIX-104
        auto& state = State();
        std::scoped_lock lk{state.mutex};
        const auto it = state.running.find(id);
        if (it->second.reported) {
            LOG_WARNING(Render_Vulkan, "DIAG-049: {} of {} done after {:.1f} s; {} MB committed",
                        what, name, Seconds(it->second.start), CommittedMb());
        }
        state.running.erase(it);
    }

private:
    struct Entry {
        std::string name;
        const char* what;
        std::chrono::steady_clock::time_point start;
        bool reported;
        double last_report = 0;
    };
    struct Shared {
        std::mutex mutex;
        std::unordered_map<u64, Entry> running;
        u64 next_id = 0;
        bool watcher_started = false;
    };
    static Shared& State() {
        static Shared state;
        return state;
    }
    static double Seconds(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    static u64 CommittedMb() {
#ifdef _WIN32
        PROCESS_MEMORY_COUNTERS_EX counters{};
        if (K32GetProcessMemoryInfo(GetCurrentProcess(),
                                    reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                                    sizeof(counters))) {
            return counters.PrivateUsage >> 20;
        }
#endif
        return 0;
    }
    static void Watch() {
        Common::SetCurrentThreadName("shadGT:BuildWatch");
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds{2});
            auto& state = State();
            std::scoped_lock lk{state.mutex};
            u32 long_running = 0;
            for (auto& [id, entry] : state.running) {
                const double seconds = Seconds(entry.start);
                long_running += seconds >= 3.0;
                if (seconds >= 3.0 && (!entry.reported || seconds - entry.last_report >= 10.0)) {
                    entry.reported = true;
                    entry.last_report = seconds;
                    LOG_WARNING(Render_Vulkan,
                                "DIAG-049: {} of {} running for {:.1f} s; {} MB committed",
                                entry.what, entry.name, Seconds(entry.start), CommittedMb());
                }
            }
            if (long_running != 0) {
                LOG_WARNING(Render_Vulkan, "DIAG-049: {} pipeline creations of {} running for 3 s "
                            "or more; {} MB committed", long_running, state.running.size(),
                            CommittedMb());
            }
#ifdef _WIN32
            // DIAG-100: a creation stuck for 30 s: once per run, every thread's stack goes to
            // user\log\stuck-stacks.txt (tools\stackdump.exe next to GTSport.exe).
            static bool dumped = false;
            const bool stuck = std::ranges::any_of(
                state.running, [](const auto& entry) { return Seconds(entry.second.start) >= 30.0; });
            if (stuck && !dumped) {
                dumped = true;
                DumpThreadStacks(L"stuck-stacks.txt");
            }
#endif
        }
    }

    std::string name;
    const char* what;
    u64 id{};
};

// FIX-057: the driver compiles a tessellated pipeline with a lot of memory, and GT Sport's race
// summary (the driver figure) needs hundreds of new ones at once. With every build worker on one,
// the process went from 11 GB to 100 GB in seconds and the driver's compiler ran out of memory.
// At most two of them are created at the same time; -DisablePerf 79 lets them all run.
class HeavyBuildGate {
public:
    explicit HeavyBuildGate(bool heavy) {
        static const bool enabled = Common::PerfFeatureEnabled(79);
        if (!heavy || !enabled) {
            return;
        }
        const auto start = std::chrono::steady_clock::now();
        Semaphore().acquire();
        held = true;
        const auto waited = std::chrono::steady_clock::now() - start;
        if (waited >= std::chrono::seconds{1}) {
            static std::atomic<u32> logged{};
            if (const u32 n = ++logged; n <= 20 || n % 100 == 0) {
                LOG_WARNING(Render_Vulkan,
                            "FIX-057: a tessellated pipeline waited {:.1f} s for its turn ({} "
                            "times)",
                            std::chrono::duration<double>(waited).count(), n);
            }
        }
    }
    ~HeavyBuildGate() {
        if (held) {
            Semaphore().release();
        }
    }

private:
    static std::counting_semaphore<64>& Semaphore() {
        static std::counting_semaphore<64> semaphore{2};
        return semaphore;
    }
    bool held = false;
};

} // namespace

using Shader::Backend::SPIRV::AuxShaderType;

static constexpr std::array LogicalStageToStageBit = {
    vk::ShaderStageFlagBits::eFragment,
    vk::ShaderStageFlagBits::eTessellationControl,
    vk::ShaderStageFlagBits::eTessellationEvaluation,
    vk::ShaderStageFlagBits::eVertex,
    vk::ShaderStageFlagBits::eGeometry,
    vk::ShaderStageFlagBits::eCompute,
};

// PERF-021: time of the last pipeline a draw needed that did not exist yet.
static std::atomic<s64> last_pipeline_miss_ms{};

// SHD-001: pipelines are not linked a second time with link-time optimization. That link is a
// full driver compile of every new pipeline again (after new content, minutes of compiling in
// the background on up to four threads, competing with the game for cores), for GPU time the
// renderer no longer lacks. RYGT_PIPELINE_LTO=1 links them as before (with counts and times).
static bool LinkTimeOptimization() {
    static const bool lto = [] {
        const char* env = std::getenv("RYGT_PIPELINE_LTO");
        return env && env[0] == '1';
    }();
    return lto;
}
static std::atomic<u64> lto_links{};
static std::atomic<u64> lto_us{};

static s64 SteadyMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void GraphicsPipeline::NotePipelineMiss() {
    last_pipeline_miss_ms.store(SteadyMs(), std::memory_order_relaxed);
}

// PERF-017: workers that link fully optimized pipelines from stage libraries, so the command
// thread only waits for the fast link.
class PipelineLinkWorkers {
public:
    static PipelineLinkWorkers& Instance() {
        static PipelineLinkWorkers workers;
        return workers;
    }

    void Push(std::function<void()>&& job) {
        {
            std::scoped_lock lk{mutex};
            jobs.push_back(std::move(job));
        }
        cv.notify_one();
    }

private:
    PipelineLinkWorkers() {
        const u32 count = std::clamp(std::thread::hardware_concurrency() / 4, 1u, 4u);
        for (u32 i = 0; i < count; ++i) {
            threads.emplace_back([this](std::stop_token stop) {
                Common::SetCurrentThreadName("shadGT:PipelineLink");
                while (true) {
                    std::function<void()> job;
                    {
                        std::unique_lock lk{mutex};
                        cv.wait(lk, stop, [this] { return !jobs.empty(); });
                        if (stop.stop_requested()) {
                            return;
                        }
                        job = std::move(jobs.front());
                        jobs.pop_front();
                    }
                    job();
                }
            });
        }
    }

    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<std::function<void()>> jobs;
    std::vector<std::jthread> threads;
};

// PERF-018: shader stage libraries (pre-rasterization and fragment shader) of runtime
// pipelines, shared by every pipeline whose stages, layout and stage state are the same. Many new
// pipelines differ from an earlier one only in blending, targets or vertex input, so they reuse
// both compiled shader libraries and only link. Libraries live until exit.
class StageLibraryCache {
public:
    static StageLibraryCache& Instance() {
        static StageLibraryCache cache;
        return cache;
    }

    vk::Pipeline Find(u64 key) {
        std::scoped_lock lk{mutex};
        const auto it = libraries.find(key);
        return it != libraries.end() ? it->second : vk::Pipeline{};
    }

    void Insert(u64 key, vk::Pipeline library) {
        std::scoped_lock lk{mutex};
        libraries.emplace(key, library);
    }

private:
    std::mutex mutex;
    std::unordered_map<u64, vk::Pipeline> libraries;
};

class LibraryKey {
public:
    explicit LibraryKey(u64 kind) {
        Add(kind);
    }
    template <typename T>
    void Add(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* bytes = reinterpret_cast<const u8*>(&value);
        data.insert(data.end(), bytes, bytes + sizeof(T));
    }
    u64 Hash() const {
        return XXH3_64bits(data.data(), data.size());
    }

private:
    std::vector<u8> data;
};

GraphicsPipeline::GraphicsPipeline(
    const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
    const Shader::Profile& profile, const GraphicsPipelineKey& key_,
    vk::PipelineCache pipeline_cache, std::span<const Shader::Info*, MaxShaderStages> infos,
    std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_infos,
    const Shader::Gcn::FetchShaderData* fetch_shader_, std::span<const vk::ShaderModule> modules,
    SerializationSupport& sdata, bool preloading)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache}, key{key_} {
    if (fetch_shader_) {
        fetch_shader = *fetch_shader_;
    }

    const vk::Device device = instance.GetDevice();
    std::ranges::copy(infos, stages.begin());
    std::ranges::copy(infos, canonical_stages.begin());
    // PERF-038: `preloading` builds a stored pipeline in the background. Its shader infos carry no
    // user data and there are no guest buffers to read: vertex inputs and the tessellation
    // helpers come from the stored serialization data; the runtime infos are the ones stored
    // with each stage's specialization, so the pipeline is the one a draw would build.
    BuildDescSetLayout(preloading);
    const auto debug_str = GetDebugString();
    BuildWatch watch{debug_str, preloading ? "stored pipeline build" : "pipeline build"};
    const bool tessellated = stages[u32(Shader::SwStage::TessellationControl)] != nullptr;
    HeavyBuildGate gate{tessellated};

    const vk::PushConstantRange push_constants = {
        .stageFlags = AllGraphicsStageBits,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };

    // SHD-002: one set per stage; with libraries, sets are independent so each shader library
    // is made with its own stages' sets only.
    const bool independent_sets = instance.IsGraphicsPipelineLibrarySupported();
    const vk::PipelineLayoutCreateInfo layout_info = {
        .flags = independent_sets ? vk::PipelineLayoutCreateFlagBits::eIndependentSetsEXT
                                  : vk::PipelineLayoutCreateFlags{},
        .setLayoutCount = num_sets,
        .pSetLayouts = set_layouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constants,
    };
    auto [layout_result, layout] = instance.GetDevice().createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess,
               "Failed to create graphics pipeline layout: {}", vk::to_string(layout_result));
    pipeline_layout = std::move(layout);
    SetObjectName(device, *pipeline_layout, "Graphics PipelineLayout {}", debug_str);

    VertexInputs<AmdGpu::Buffer> guest_buffers;
    if (!preloading && !instance.IsVertexInputDynamicState()) {
        const auto& vs_info = runtime_infos[u32(Shader::SwStage::Vertex)].sw.vs;
        GetVertexInputs(sdata.vertex_attributes, sdata.vertex_bindings, sdata.divisors,
                        guest_buffers, vs_info.step_rate_0, vs_info.step_rate_1);
    }

    const vk::PipelineVertexInputDivisorStateCreateInfo divisor_state = {
        .vertexBindingDivisorCount = static_cast<u32>(sdata.divisors.size()),
        .pVertexBindingDivisors = sdata.divisors.data(),
    };

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info = {
        .pNext = sdata.divisors.empty() ? nullptr : &divisor_state,
        .vertexBindingDescriptionCount = static_cast<u32>(sdata.vertex_bindings.size()),
        .pVertexBindingDescriptions = sdata.vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(sdata.vertex_attributes.size()),
        .pVertexAttributeDescriptions = sdata.vertex_attributes.data(),
    };

    const auto topology = LiverpoolToVK::PrimitiveType(key.prim_type);
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly = {
        .topology = topology,
    };

    const bool is_rect_list = key.prim_type == AmdGpu::PrimitiveType::RectList;
    const bool is_quad_list = key.prim_type == AmdGpu::PrimitiveType::QuadList;
    const vk::PipelineTessellationStateCreateInfo tessellation_state = {
        .patchControlPoints = is_rect_list ? 3U : (is_quad_list ? 4U : key.patch_control_points),
    };

    vk::StructureChain raster_chain = {
        vk::PipelineRasterizationStateCreateInfo{
            .depthClampEnable = key.depth_clamp_enable &&
                                (!key.depth_clip_enable || instance.IsDepthClipEnableSupported()),
            .rasterizerDiscardEnable = false,
            .polygonMode = LiverpoolToVK::PolygonMode(key.polygon_mode),
            .lineWidth = 1.0f,
        },
        vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT{
            .provokingVertexMode = key.provoking_vtx_last == AmdGpu::ProvokingVtxLast::First
                                       ? vk::ProvokingVertexModeEXT::eFirstVertex
                                       : vk::ProvokingVertexModeEXT::eLastVertex,
        },
        vk::PipelineRasterizationDepthClipStateCreateInfoEXT{
            .depthClipEnable = key.depth_clip_enable,
        },
    };

    if (!instance.IsProvokingVertexSupported()) {
        raster_chain.unlink<vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT>();
    }
    if (!instance.IsDepthClipEnableSupported()) {
        raster_chain.unlink<vk::PipelineRasterizationDepthClipStateCreateInfoEXT>();
    }

    const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
    sdata.multisampling = {
        .rasterizationSamples = LiverpoolToVK::NumSamples(
            key.num_samples, instance.GetColorSampleCounts() & instance.GetDepthSampleCounts()),
        .sampleShadingEnable =
            fs_info.addr_flags.persp_sample_ena || fs_info.addr_flags.linear_sample_ena,
    };

    const vk::PipelineViewportDepthClipControlCreateInfoEXT clip_control = {
        .negativeOneToOne = key.clip_space == AmdGpu::ClipSpace::MinusWToW,
    };

    const vk::PipelineViewportStateCreateInfo viewport_info = {
        .pNext = instance.IsDepthClipControlSupported() ? &clip_control : nullptr,
    };

    boost::container::static_vector<vk::DynamicState, 32> dynamic_states = {
        vk::DynamicState::eViewportWithCount,  vk::DynamicState::eScissorWithCount,
        vk::DynamicState::eBlendConstants,     vk::DynamicState::eDepthTestEnable,
        vk::DynamicState::eDepthWriteEnable,   vk::DynamicState::eDepthCompareOp,
        vk::DynamicState::eDepthBiasEnable,    vk::DynamicState::eDepthBias,
        vk::DynamicState::eStencilTestEnable,  vk::DynamicState::eStencilReference,
        vk::DynamicState::eStencilCompareMask, vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eStencilOp,          vk::DynamicState::eCullMode,
        vk::DynamicState::eFrontFace,          vk::DynamicState::eRasterizerDiscardEnable,
        vk::DynamicState::eLineWidth,          vk::DynamicState::ePrimitiveRestartEnable,
    };

    if (instance.IsDepthBoundsSupported()) {
        dynamic_states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
        dynamic_states.push_back(vk::DynamicState::eDepthBounds);
    }
    if (instance.IsDynamicColorWriteMaskSupported()) {
        dynamic_states.push_back(vk::DynamicState::eColorWriteMaskEXT);
    }
    if (instance.IsVertexInputDynamicState()) {
        dynamic_states.push_back(vk::DynamicState::eVertexInputEXT);
    } else if (!sdata.vertex_bindings.empty()) {
        dynamic_states.push_back(vk::DynamicState::eVertexInputBindingStride);
    }

    const vk::PipelineDynamicStateCreateInfo dynamic_info = {
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    boost::container::static_vector<vk::PipelineShaderStageCreateInfo, MaxShaderStages>
        shader_stages;
    auto stage = u32(Shader::SwStage::Vertex);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = modules[stage],
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::Geometry);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eGeometry,
            .module = modules[stage],
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::TessellationControl);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (is_rect_list || is_quad_list) {
        const auto type = is_quad_list ? AuxShaderType::QuadListTCS : AuxShaderType::RectListTCS;
        const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
        sdata.tcs = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
            type, fs_info, *infos[u32(Shader::SwStage::Vertex)],
            infos[u32(Shader::SwStage::Fragment)]);
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationControl,
            .module = CompileSPV(sdata.tcs, instance.GetDevice()),
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::TessellationEval);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (is_rect_list || is_quad_list) {
        const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;
        sdata.tes = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
            AuxShaderType::PassthroughTES, fs_info, *infos[u32(Shader::SwStage::Vertex)],
            infos[u32(Shader::SwStage::Fragment)]);
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eTessellationEvaluation,
            .module = CompileSPV(sdata.tes, instance.GetDevice()),
            .pName = "main",
        });
    }
    stage = u32(Shader::SwStage::Fragment);
    if (infos[stage]) {
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = modules[stage],
            .pName = "main",
        });
    } else if (runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs.clip_distance_emulation) {
        const auto& vs = runtime_infos[static_cast<u32>(Shader::SwStage::Vertex)].hw.vs;

        sdata.fragment = Shader::Backend::SPIRV::EmitDiscardFragmentShader(vs.outputs);
        shader_stages.emplace_back(vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = CompileSPV(sdata.fragment, instance.GetDevice()),
            .pName = "main",
        });
    }

    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
        .requiredSubgroupSize = 64,
    };
    for (auto& stage : shader_stages) {
        stage.pNext = instance.IsSubgroupSize64Supported() ? &subgroup_size_ci : nullptr;
    }

    const auto depth_format =
        instance.GetSupportedFormat(LiverpoolToVK::DepthFormat(key.z_format, key.stencil_format),
                                    vk::FormatFeatureFlagBits2::eDepthStencilAttachment);
    std::array<vk::Format, Shader::IR::NumRenderTargets> color_formats;
    for (s32 i = 0; i < key.num_color_attachments; ++i) {
        const auto& col_buf = key.color_buffers[i];
        const auto format = LiverpoolToVK::SurfaceFormat(col_buf.data_format, col_buf.num_format);
        const auto color_format =
            instance.GetSupportedFormat(format, vk::FormatFeatureFlagBits2::eColorAttachment);
        if (!instance.IsFormatSupported(color_format,
                                        vk::FormatFeatureFlagBits2::eColorAttachment)) {
            LOG_WARNING(Render_Vulkan,
                        "color buffer format {} does not support COLOR_ATTACHMENT_BIT",
                        vk::to_string(color_format));
        }
        color_formats[i] = color_format;
    }

    std::array<vk::SampleCountFlagBits, AmdGpu::NUM_COLOR_BUFFERS> color_samples;
    std::ranges::transform(key.color_samples, color_samples.begin(), [&instance](u8 num_samples) {
        return num_samples ? LiverpoolToVK::NumSamples(num_samples, instance.GetColorSampleCounts())
                           : vk::SampleCountFlagBits::e1;
    });
    const vk::AttachmentSampleCountInfoAMD mixed_samples = {
        .colorAttachmentCount = key.num_color_attachments,
        .pColorAttachmentSamples = color_samples.data(),
        .depthStencilAttachmentSamples =
            LiverpoolToVK::NumSamples(key.depth_samples, instance.GetDepthSampleCounts()),
    };

    const vk::PipelineRenderingCreateInfo pipeline_rendering_ci = {
        .pNext = instance.IsMixedDepthSamplesSupported() ? &mixed_samples : nullptr,
        .colorAttachmentCount = key.num_color_attachments,
        .pColorAttachmentFormats = color_formats.data(),
        .depthAttachmentFormat = key.z_format != AmdGpu::DepthBuffer::ZFormat::Invalid
                                     ? depth_format
                                     : vk::Format::eUndefined,
        .stencilAttachmentFormat = key.stencil_format != AmdGpu::DepthBuffer::StencilFormat::Invalid
                                       ? depth_format
                                       : vk::Format::eUndefined,
    };

    std::array<vk::PipelineColorBlendAttachmentState, AmdGpu::NUM_COLOR_BUFFERS> attachments;
    for (u32 i = 0; i < key.num_color_attachments; i++) {
        const auto& control = key.blend_controls[i];

        const auto src_color = LiverpoolToVK::BlendFactor(control.color_src_factor);
        const auto dst_color = LiverpoolToVK::BlendFactor(control.color_dst_factor);
        const auto color_blend = LiverpoolToVK::BlendOp(control.color_func);

        const auto src_alpha = control.separate_alpha_blend
                                   ? LiverpoolToVK::BlendFactor(control.alpha_src_factor)
                                   : src_color;
        const auto dst_alpha = control.separate_alpha_blend
                                   ? LiverpoolToVK::BlendFactor(control.alpha_dst_factor)
                                   : dst_color;
        const auto alpha_blend =
            control.separate_alpha_blend ? LiverpoolToVK::BlendOp(control.alpha_func) : color_blend;

        // Vulkan ignores blend factors for min/max, but a factor that zeroes one operand
        // makes the operation collapse to a plain selection: min(s, 0) is 0 and max(s, 0)
        // is s for normalized alpha. Rewrite those to the equivalent add so the result is
        // exact instead of leaving the other operand to survive.
        auto eff_src_alpha = src_alpha;
        auto eff_dst_alpha = dst_alpha;
        auto eff_alpha_blend = alpha_blend;
        if (alpha_blend == vk::BlendOp::eMin || alpha_blend == vk::BlendOp::eMax) {
            const bool takes_max = alpha_blend == vk::BlendOp::eMax;
            if (src_alpha == vk::BlendFactor::eOne && dst_alpha == vk::BlendFactor::eZero) {
                eff_alpha_blend = vk::BlendOp::eAdd;
                eff_src_alpha = takes_max ? vk::BlendFactor::eOne : vk::BlendFactor::eZero;
                eff_dst_alpha = vk::BlendFactor::eZero;
            } else if (src_alpha == vk::BlendFactor::eZero && dst_alpha == vk::BlendFactor::eOne) {
                eff_alpha_blend = vk::BlendOp::eAdd;
                eff_src_alpha = vk::BlendFactor::eZero;
                eff_dst_alpha = takes_max ? vk::BlendFactor::eOne : vk::BlendFactor::eZero;
            }
        }

        const auto color_scaled_min_max =
            (color_blend == vk::BlendOp::eMin || color_blend == vk::BlendOp::eMax) &&
            (src_color != vk::BlendFactor::eOne || dst_color != vk::BlendFactor::eOne) &&
            !key.color_buffers[i].blend_self_scale;
        const auto alpha_scaled_min_max =
            (eff_alpha_blend == vk::BlendOp::eMin || eff_alpha_blend == vk::BlendOp::eMax) &&
            (eff_src_alpha != vk::BlendFactor::eOne || eff_dst_alpha != vk::BlendFactor::eOne);
        if (color_scaled_min_max || alpha_scaled_min_max) {
            LOG_WARNING(
                Render_Vulkan,
                "Unimplemented use of min/max blend op with blend factor not equal to one.");
        }

        attachments[i] = vk::PipelineColorBlendAttachmentState{
            .blendEnable = control.enable,
            .srcColorBlendFactor = src_color,
            .dstColorBlendFactor = dst_color,
            .colorBlendOp = color_blend,
            .srcAlphaBlendFactor = eff_src_alpha,
            .dstAlphaBlendFactor = eff_dst_alpha,
            .alphaBlendOp = eff_alpha_blend,
            .colorWriteMask =
                instance.IsDynamicColorWriteMaskSupported()
                    ? vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
                    : key.write_masks[i],
        };

        // The shader squares its color output for this attachment (see PsColorBuffer), so the
        // factors must not scale the operands again.
        if (key.color_buffers[i].blend_self_scale) {
            LOG_WARNING(
                Render_Vulkan,
                "Emulating scaled min/max blend with squared shader output on attachment {}", i);
            attachments[i].srcColorBlendFactor = vk::BlendFactor::eOne;
            attachments[i].dstColorBlendFactor = vk::BlendFactor::eOne;
        }

        // On GCN GPU there is an additional mask which allows to control color components exported
        // from a pixel shader. A situation possible, when the game may mask out the alpha channel,
        // while it is still need to be used in blending ops. For such cases, HW will default alpha
        // to 1 and perform the blending, while shader normally outputs 0 in the last component.
        // Unfortunatelly, Vulkan doesn't provide any control on blend inputs, so below we detecting
        // such cases and override alpha value in order to emulate HW behaviour.
        const auto has_alpha_masked_out =
            (key.cb_shader_mask.GetMask(i) & AmdGpu::ColorBufferMask::ComponentA) == 0;
        const auto has_src_alpha_in_src_blend = src_color == vk::BlendFactor::eSrcAlpha ||
                                                src_color == vk::BlendFactor::eOneMinusSrcAlpha;
        const auto has_src_alpha_in_dst_blend = dst_color == vk::BlendFactor::eSrcAlpha ||
                                                dst_color == vk::BlendFactor::eOneMinusSrcAlpha;
        if (has_alpha_masked_out && has_src_alpha_in_src_blend) {
            attachments[i].srcColorBlendFactor = src_color == vk::BlendFactor::eSrcAlpha
                                                     ? vk::BlendFactor::eOne
                                                     : vk::BlendFactor::eZero; // 1-A
        }
        if (has_alpha_masked_out && has_src_alpha_in_dst_blend) {
            attachments[i].dstColorBlendFactor = dst_color == vk::BlendFactor::eSrcAlpha
                                                     ? vk::BlendFactor::eOne
                                                     : vk::BlendFactor::eZero; // 1-A
        }

        if (key.color_buffers[i].blend_swizzled_alpha) {
            LiverpoolToVK::SetSwizzledAlphaBlend(attachments[i]);
            static const bool logged = [] {
                LOG_INFO(Render_Vulkan, "Using dual-source swizzled source-alpha blend emulation");
                return true;
            }();
        } else if (key.color_buffers[i].blend_swizzled_factors) {
            // Both functions match for this emulation.
            LiverpoolToVK::SetSwizzledFactorBlend(attachments[i], control.color_func);
            static const bool logged = [] {
                LOG_INFO(Render_Vulkan, "Using dual-source swizzled factor blend emulation");
                return true;
            }();
        }
    }

    const vk::PipelineColorBlendStateCreateInfo color_blending = {
        .logicOpEnable =
            instance.IsLogicOpSupported() && key.logic_op != AmdGpu::ColorControl::LogicOp::Copy,
        .logicOp = LiverpoolToVK::LogicOp(key.logic_op),
        .attachmentCount = key.num_color_attachments,
        .pAttachments = attachments.data(),
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };

    // Required by spec unless VK_EXT_extended_dynamic_state3 is supported.
    // In practice, we use dynamic state for all of it.
    constexpr vk::PipelineDepthStencilStateCreateInfo depth_stencil_info = {};

    // FIX-057: the driver's optimizer is what ran out of memory on GT Sport's tessellated
    // pipelines (also whole, two at a time); they are compiled without optimization.
    // -DisablePerf 81 optimizes them.
    static const bool plain_tessellated = Common::PerfFeatureEnabled(81);
    const vk::GraphicsPipelineCreateInfo pipeline_info = {
        .pNext = &pipeline_rendering_ci,
        .flags = tessellated && plain_tessellated ? vk::PipelineCreateFlagBits::eDisableOptimization
                                                  : vk::PipelineCreateFlags{},
        .stageCount = static_cast<u32>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = !instance.IsVertexInputDynamicState() ? &vertex_input_info : nullptr,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = &tessellation_state,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_chain.get(),
        .pMultisampleState = &sdata.multisampling,
        .pDepthStencilState =
            !instance.IsExtendedDynamicState3Supported() ? &depth_stencil_info : nullptr,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_info,
        .layout = *pipeline_layout,
    };

    const auto create_start = std::chrono::steady_clock::now();
    // PERF-038: a background build compiles the whole pipeline at once (no libraries to link
    // and optimize again later).
    // FIX-057: tessellated pipelines are compiled whole, not from libraries: on GT Sport's race
    // summary their library compiles (with link-time optimization info kept) took the process
    // from 11 GB to over 50 GB with only two running. -DisablePerf 80 builds them from libraries.
    static const bool whole_tessellated = Common::PerfFeatureEnabled(80);
    // DIAG-058 (SHADGT_NO_LIBRARIES=1): every pipeline is compiled whole, never fast-linked from
    // libraries (does the summary's driver figure render differently from fast-linked ones?).
    static const bool no_libraries = [] {
        const char* env = std::getenv("SHADGT_NO_LIBRARIES");
        return env && env[0] == '1';
    }();
    if (!preloading && !no_libraries && instance.IsGraphicsPipelineLibrarySupported() &&
        !(tessellated && whole_tessellated)) {
        // PERF-017: a pipeline first needed during play is built from four separately compiled
        // libraries and linked without cross-stage optimization, which the driver does in a
        // fraction of a full compile. A worker then links the same libraries with link-time
        // optimization, and draws switch to that pipeline once it exists. Both pipelines come
        // from the same create info, so they render the same.
        library_link = std::make_unique<LibraryLink>();
        boost::container::static_vector<vk::PipelineShaderStageCreateInfo, MaxShaderStages>
            pre_raster_stages;
        boost::container::static_vector<vk::PipelineShaderStageCreateInfo, 1> fragment_stages;
        for (const auto& stage_info : shader_stages) {
            if (stage_info.stage == vk::ShaderStageFlagBits::eFragment) {
                fragment_stages.push_back(stage_info);
            } else {
                pre_raster_stages.push_back(stage_info);
            }
        }
        // SHD-002: the shader libraries' layouts hold their own stages' sets only (the others
        // are null), so a library doesn't depend on the other stages' resources.
        const auto library_layout = [&](bool fragment) {
            std::array<vk::DescriptorSetLayout, Shader::Backend::MaxDescriptorSets> sets{};
            u32 count = 0;
            for (u32 set = 0; set < num_sets; ++set) {
                const bool fragment_set = set == Shader::Backend::StageSet(
                                                     u32(Shader::SwStage::Fragment));
                if (fragment_set == fragment && set_layout_hashes[set] != 0) {
                    sets[set] = set_layouts[set];
                    count = set + 1;
                }
            }
            auto [result, created] = device.createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo{
                .flags = vk::PipelineLayoutCreateFlagBits::eIndependentSetsEXT,
                .setLayoutCount = count,
                .pSetLayouts = sets.data(),
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &push_constants,
            });
            ASSERT_MSG(result == vk::Result::eSuccess, "Failed to create a library layout: {}",
                       vk::to_string(result));
            return std::move(created);
        };
        const vk::UniquePipelineLayout pre_raster_layout = library_layout(false);
        const vk::UniquePipelineLayout fragment_layout = library_layout(true);
        const std::array<vk::GraphicsPipelineLibraryFlagsEXT, 4> parts = {
            vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface,
            vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders,
            vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader,
            vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface,
        };
        std::array<vk::GraphicsPipelineLibraryCreateInfoEXT, 4> library_infos{};
        std::array<vk::GraphicsPipelineCreateInfo, 4> part_infos{};
        for (size_t i = 0; i < parts.size(); ++i) {
            library_infos[i] = vk::GraphicsPipelineLibraryCreateInfoEXT{
                .pNext = &pipeline_rendering_ci,
                .flags = parts[i],
            };
            auto& part_info = part_infos[i];
            part_info = pipeline_info;
            part_info.pNext = &library_infos[i];
            part_info.flags = vk::PipelineCreateFlagBits::eLibraryKHR;
            if (LinkTimeOptimization()) { // SHD-001
                part_info.flags |= vk::PipelineCreateFlagBits::eRetainLinkTimeOptimizationInfoEXT;
            }
            part_info.stageCount = 0;
            part_info.pStages = nullptr;
            if (i == 1) {
                part_info.stageCount = static_cast<u32>(pre_raster_stages.size());
                part_info.pStages = pre_raster_stages.data();
                part_info.layout = *pre_raster_layout; // SHD-002
            } else if (i == 2) {
                part_info.stageCount = static_cast<u32>(fragment_stages.size());
                part_info.pStages = fragment_stages.data();
                part_info.layout = *fragment_layout; // SHD-002
            }
        }
        const auto create_library = [&](size_t i) {
            BuildWatch part_watch{debug_str, i == 1 ? "pre-raster library"
                                             : i == 2 ? "fragment library"
                                                      : "interface library"};
            auto [library_result, library] =
                device.createGraphicsPipeline(pipeline_cache, part_infos[i]);
            ASSERT_MSG(library_result == vk::Result::eSuccess,
                       "Failed to create graphics pipeline library: {}",
                       vk::to_string(library_result));
            return library;
        };

        // PERF-018: the shader libraries are keyed by everything their create info reads: the
        // layout's bindings, the stage modules, and the state of their pipeline subset.
        LibraryKey pre_raster_key{1};
        // SHD-002: the sets of the pre-rasterization stages only.
        for (u32 set = 0; set < num_sets; ++set) {
            if (set != Shader::Backend::StageSet(u32(Shader::SwStage::Fragment))) {
                pre_raster_key.Add(set_layout_hashes[set]);
            }
        }
        for (const auto& stage_info : pre_raster_stages) {
            pre_raster_key.Add(stage_info.stage);
            pre_raster_key.Add(static_cast<VkShaderModule>(stage_info.module));
        }
        pre_raster_key.Add(tessellation_state.patchControlPoints);
        pre_raster_key.Add(
            raster_chain.get<vk::PipelineRasterizationStateCreateInfo>().depthClampEnable);
        pre_raster_key.Add(key.polygon_mode);
        pre_raster_key.Add(key.provoking_vtx_last);
        pre_raster_key.Add(key.depth_clip_enable);
        pre_raster_key.Add(key.clip_space);
        for (const auto state : dynamic_states) {
            pre_raster_key.Add(state);
        }

        LibraryKey fragment_key{2};
        // SHD-002: the fragment stage's set only.
        fragment_key.Add(set_layout_hashes[Shader::Backend::StageSet(u32(Shader::SwStage::Fragment))]);
        for (const auto& stage_info : fragment_stages) {
            fragment_key.Add(static_cast<VkShaderModule>(stage_info.module));
        }
        // SHD-004: a fragment shader library doesn't depend on the attachments: of
        // VkPipelineRenderingCreateInfo only the view mask is part of its state (formats are
        // ignored), and the multisample state only with sample shading (GPL's fragment shader
        // subset). One library serves a pixel shader in every pass it draws in.
        fragment_key.Add(sdata.multisampling.sampleShadingEnable);
        if (sdata.multisampling.sampleShadingEnable) {
            fragment_key.Add(sdata.multisampling.rasterizationSamples);
            fragment_key.Add(sdata.multisampling.minSampleShading);
        }
        fragment_key.Add(pipeline_rendering_ci.viewMask);
        for (const auto state : dynamic_states) {
            fragment_key.Add(state);
        }

        auto& library_cache = StageLibraryCache::Instance();
        const u64 pre_raster_hash = pre_raster_key.Hash();
        const u64 fragment_hash = fragment_key.Hash();
        // -DisablePerf 18 compiles every library for every pipeline, one after another.
        const bool share_libraries = Common::PerfFeatureEnabled(18);
        vk::Pipeline pre_raster_library =
            share_libraries ? library_cache.Find(pre_raster_hash) : vk::Pipeline{};
        vk::Pipeline fragment_library =
            share_libraries ? library_cache.Find(fragment_hash) : vk::Pipeline{};
        const bool pre_raster_reused = bool(pre_raster_library);
        const bool fragment_reused = bool(fragment_library);
        // PERF-018: the two shader libraries are compiled at the same time, the fragment one
        // on another thread.
        std::future<vk::Pipeline> fragment_future;
        if (!fragment_library && share_libraries) {
            fragment_future = std::async(std::launch::async, create_library, size_t{2});
        }
        std::array<vk::Pipeline, 4> library_handles{};
        library_handles[0] = create_library(0);
        if (!pre_raster_library) {
            pre_raster_library = create_library(1);
            library_cache.Insert(pre_raster_hash, pre_raster_library);
        }
        library_handles[3] = create_library(3);
        if (fragment_future.valid()) {
            fragment_library = fragment_future.get();
        } else if (!fragment_library) {
            fragment_library = create_library(2);
        }
        if (!fragment_reused) {
            library_cache.Insert(fragment_hash, fragment_library);
        }
        library_handles[1] = pre_raster_library;
        library_handles[2] = fragment_library;
        library_link->owned_libraries[0] = vk::UniquePipeline{library_handles[0], device};
        library_link->owned_libraries[1] = vk::UniquePipeline{library_handles[3], device};
        const auto libraries_done = std::chrono::steady_clock::now();

        const vk::PipelineLibraryCreateInfoKHR link_libraries = {
            .libraryCount = static_cast<u32>(library_handles.size()),
            .pLibraries = library_handles.data(),
        };
        const vk::GraphicsPipelineCreateInfo link_info = {
            .pNext = &link_libraries,
            .layout = *pipeline_layout,
        };
        auto [link_result, linked] = device.createGraphicsPipelineUnique(pipeline_cache, link_info);
        ASSERT_MSG(link_result == vk::Result::eSuccess, "Failed to link graphics pipeline: {}",
                   vk::to_string(link_result));
        pipeline = std::move(linked);
        SetObjectName(device, *pipeline, "Graphics Pipeline {}", debug_str);
        LOG_WARNING(
            Render_Vulkan,
            "Graphics pipeline {}: libraries {:.1f} ms (pre-raster {}, fragment {}), fast "
            "link {:.1f} ms",
            debug_str,
            std::chrono::duration<double, std::milli>(libraries_done - create_start).count(),
            pre_raster_reused ? "reused" : "compiled", fragment_reused ? "reused" : "compiled",
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                      libraries_done)
                .count());

        if (!LinkTimeOptimization()) {
            return; // SHD-001: the fast-linked pipeline is the one drawn with
        }
        auto promise = std::make_shared<std::promise<void>>();
        library_link->optimized = promise->get_future();
        const auto layout = *pipeline_layout;
        PipelineLinkWorkers::Instance().Push([this, device, layout, library_handles, promise,
                                              cache = pipeline_cache, debug_str, tessellated] {
            // PERF-021: the fast-linked pipeline renders the same, so the optimized one can
            // wait until new pipelines stop appearing (the worker only sleeps meanwhile).
            if (Common::PerfFeatureEnabled(21)) {
                while (SteadyMs() - last_pipeline_miss_ms.load(std::memory_order_relaxed) < 3000) {
                    std::this_thread::sleep_for(std::chrono::milliseconds{100});
                }
            }
            const vk::PipelineLibraryCreateInfoKHR libraries_info = {
                .libraryCount = static_cast<u32>(library_handles.size()),
                .pLibraries = library_handles.data(),
            };
            const vk::GraphicsPipelineCreateInfo optimized_info = {
                .pNext = &libraries_info,
                .flags = vk::PipelineCreateFlagBits::eLinkTimeOptimizationEXT,
                .layout = layout,
            };
            BuildWatch watch{debug_str, "optimized link"};
            HeavyBuildGate gate{tessellated};
            const auto link_start = std::chrono::steady_clock::now();
            auto [result, optimized] = device.createGraphicsPipeline(cache, optimized_info);
            if (result == vk::Result::eSuccess) {
                optimized_pipeline.store(optimized, std::memory_order_release);
            }
            // SHD-001: how much compiling the optimized links cost.
            const u64 us = std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - link_start)
                               .count();
            const u64 total_us = lto_us.fetch_add(us) + us;
            if (const u64 links = ++lto_links; links % 200 == 0) {
                LOG_WARNING(Render_Vulkan,
                            "SHD-001: {} optimized links done, {:.1f} s of driver compiling",
                            links, total_us / 1e6);
            }
            promise->set_value();
        });
        return;
    }

    std::optional<BuildWatch> whole_watch;
    if (!preloading) {
        whole_watch.emplace(debug_str, "whole pipeline compile");
    }
    auto [pipeline_result, pipe] =
        device.createGraphicsPipelineUnique(pipeline_cache, pipeline_info);
    whole_watch.reset();
    // PERF-DIAG-013: driver compile time of graphics pipelines created at runtime.
    if (!preloading) {
        LOG_WARNING(Render_Vulkan, "Graphics pipeline {}: driver create {:.1f} ms", debug_str,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              create_start)
                        .count());
    }
    ASSERT_MSG(pipeline_result == vk::Result::eSuccess, "Failed to create graphics pipeline: {}",
               vk::to_string(pipeline_result));
    pipeline = std::move(pipe);
    SetObjectName(device, *pipeline, "Graphics Pipeline {}", debug_str);
}

GraphicsPipeline::~GraphicsPipeline() {
    if (library_link && library_link->optimized.valid()) {
        library_link->optimized.wait();
    }
    if (const VkPipeline optimized = optimized_pipeline.exchange(VK_NULL_HANDLE)) {
        instance.GetDevice().destroyPipeline(optimized);
    }
}

template <typename Attribute, typename Binding>
void GraphicsPipeline::GetVertexInputs(
    VertexInputs<Attribute>& attributes, VertexInputs<Binding>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const {
    using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
    if (fetch_shader.Empty() || fetch_shader.attributes.empty()) {
        return;
    }
    const auto& vs_info = GetStage(Shader::SwStage::Vertex);
    for (u32 semantic = 0; semantic < fetch_shader.attributes.size(); ++semantic) {
        const auto& attrib = fetch_shader.attributes[semantic];
        const auto step_rate = attrib.GetStepRate();
        const auto buffer = attrib.GetSharp(vs_info);
        auto format = LiverpoolToVK::TrySurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        if (format == vk::Format::eUndefined) {
            // FIX-052: a garbage V# (data format 13 with number format 1 in a GT Sport race)
            // asserted and stopped the game. Its attribute reads four floats instead; the draw's
            // data is garbage either way.
            static std::atomic<u32> logged{};
            if (logged++ < 10) {
                LOG_WARNING(Render_Vulkan,
                            "FIX-052: vertex attribute {} has no format (data {}, number {})",
                            semantic, static_cast<u32>(buffer.GetDataFmt()),
                            static_cast<u32>(buffer.GetNumberFmt()));
            }
            format = vk::Format::eR32G32B32A32Sfloat;
        }
        attributes.push_back(Attribute{
            .location = semantic,
            .binding = semantic,
            .format = format,
            .offset = 0,
        });
        bindings.push_back(Binding{
            .binding = semantic,
            .stride = buffer.GetStride(),
            .inputRate = step_rate == InstanceIdType::None ? vk::VertexInputRate::eVertex
                                                           : vk::VertexInputRate::eInstance,
        });
        const u32 divisor = step_rate == InstanceIdType::OverStepRate0
                                ? step_rate_0
                                : (step_rate == InstanceIdType::OverStepRate1 ? step_rate_1 : 1);
        if constexpr (std::is_same_v<Binding, vk::VertexInputBindingDescription2EXT>) {
            bindings.back().divisor = divisor;
        } else if (step_rate != InstanceIdType::None) {
            divisors.push_back(vk::VertexInputBindingDivisorDescriptionEXT{
                .binding = semantic,
                .divisor = divisor,
            });
        }
        guest_buffers.emplace_back(buffer);
    }
}

// Declare templated GetVertexInputs for necessary types.
template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription>& attributes,
    VertexInputs<vk::VertexInputBindingDescription>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const;
template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription2EXT>& attributes,
    VertexInputs<vk::VertexInputBindingDescription2EXT>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<AmdGpu::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1) const;

void GraphicsPipeline::BuildDescSetLayout(bool preloading) {
    // SHD-002: one set per stage, its bindings numbered from 0.
    std::array<boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32>,
               Shader::Backend::MaxDescriptorSets>
        set_bindings;
    std::array<u32, Shader::Backend::MaxDescriptorSets> set_descriptors{};

    for (const auto* stage : stages) {
        if (!stage) {
            continue;
        }
        const u32 set = Shader::Backend::StageSet(u32(stage->sw_stage));
        auto& bindings = set_bindings[set];
        u32 binding{};
        if (const u32 sw = u32(stage->sw_stage);
            sw < Shader::Backend::StageBufferOffsetCount.size() &&
            stage->buffers.size() > Shader::Backend::StageBufferOffsetCount[sw]) {
            static std::atomic<u32> logged{};
            if (logged++ < 10) {
                LOG_ERROR(Render_Vulkan,
                          "SHD-002: {}_{:#x} has {} buffers, more than its {} push offsets",
                          stage->hw_stage, stage->pgm_hash, stage->buffers.size(),
                          Shader::Backend::StageBufferOffsetCount[sw]);
            }
        }
        const auto stage_bit = LogicalStageToStageBit[u32(stage->sw_stage)];
        for (const auto& buffer : stage->buffers) {
            if (!preloading) {
                [[maybe_unused]] const auto sharp = buffer.GetSharp(*stage);
            }
            bindings.push_back({
                .binding = binding++,
                .descriptorType = vk::DescriptorType::eStorageBuffer,
                .descriptorCount = 1,
                .stageFlags = stage_bit,
            });
        }
        for (const auto& image : stage->images) {
            const u32 num_bindings = image.NumBindings(*stage);
            bindings.push_back({
                .binding = binding,
                .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage
                                                   : vk::DescriptorType::eSampledImage,
                .descriptorCount = num_bindings,
                .stageFlags = stage_bit,
            });
            binding += num_bindings;
        }
        for (const auto& sampler : stage->samplers) {
            bindings.push_back({
                .binding = binding++,
                .descriptorType = vk::DescriptorType::eSampler,
                .descriptorCount = 1,
                .stageFlags = stage_bit,
            });
        }
        set_descriptors[set] = binding;
    }
    // The fragment set is pushed when it fits, else the vertex set; the others are allocated.
    push_set = NoPushSet;
    for (const auto sw : {Shader::SwStage::Fragment, Shader::SwStage::Vertex}) {
        const u32 set = Shader::Backend::StageSet(u32(sw));
        if (!set_bindings[set].empty() && set_descriptors[set] < instance.MaxPushDescriptors()) {
            push_set = set;
            break;
        }
    }
    const vk::Device device = instance.GetDevice();
    num_sets = 0;
    set_layout_hashes = {};
    for (u32 set = 0; set < set_bindings.size(); ++set) {
        auto& bindings = set_bindings[set];
        if (bindings.empty()) {
            continue;
        }
        const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
            .flags = set == push_set ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                                     : vk::DescriptorSetLayoutCreateFlagBits{},
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        };
        set_layout_hashes[set] =
            XXH3_64bits(bindings.data(), bindings.size() * sizeof(bindings[0])) ^
            (set == push_set ? 0x9E3779B97F4A7C15ULL : 0) ^ (u64(set) << 56) | 1;
        auto [layout_result, layout] = device.createDescriptorSetLayoutUnique(desc_layout_ci);
        ASSERT_MSG(layout_result == vk::Result::eSuccess,
                   "Failed to create graphics descriptor set layout: {}",
                   vk::to_string(layout_result));
        desc_layouts[set] = std::move(layout);
        set_layouts[set] = *desc_layouts[set];
        num_sets = set + 1;
    }
    // Sets between stage sets: an empty layout.
    for (u32 set = 0; set < num_sets; ++set) {
        if (!set_layouts[set]) {
            if (!empty_set_layout) {
                auto [result, layout] =
                    device.createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo{});
                ASSERT_MSG(result == vk::Result::eSuccess, "Failed to create an empty layout: {}",
                           vk::to_string(result));
                empty_set_layout = std::move(layout);
            }
            set_layouts[set] = *empty_set_layout;
        }
    }
}

} // namespace Vulkan
