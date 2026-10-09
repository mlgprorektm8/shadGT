// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <imgui.h>

#include "common/assert.h"
#include "common/native_clock.h"
#include "common/singleton.h"
#include "core/memory.h"
#include "core/signals.h"
#include "debug_state.h"
#include "devtools/widget/common.h"
#include "libraries/kernel/time.h"
#include "libraries/system/msgdialog.h"
#include "video_core/amdgpu/pm4_cmds.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

using namespace DebugStateType;

DebugStateImpl& DebugState = *Common::Singleton<DebugStateImpl>::Instance();

bool DebugStateType::showing_debug_menu_bar = false;

static ThreadID ThisThreadID() {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return pthread_self();
#endif
}

static void PauseThread(ThreadID id) {
#ifdef _WIN32
    auto handle = OpenThread(THREAD_SUSPEND_RESUME, FALSE, id);
    SuspendThread(handle);
    CloseHandle(handle);
#else
    pthread_kill(id, SIGSLEEP);
#endif
}

static void ResumeThread(ThreadID id) {
#ifdef _WIN32
    auto handle = OpenThread(THREAD_SUSPEND_RESUME, FALSE, id);
    ResumeThread(handle);
    CloseHandle(handle);
#else
    pthread_kill(id, SIGSLEEP);
#endif
}

std::string DebugStateImpl::DescribeGuestThreads() {
#ifdef _WIN32
    std::vector<ThreadID> ids;
    {
        std::lock_guard lock{guest_threads_mutex};
        ids = guest_threads;
    }
    auto* memory = Core::Memory::Instance();
    const auto describe = [&](u64 address) {
        if (auto host = Core::DescribeHostAddress(address); !host.empty()) {
            return "host " + host;
        }
        return memory->DescribeAddress(address);
    };
    std::string out;
    for (const ThreadID id : ids) {
        if (id == ThisThreadID()) {
            continue;
        }
        const HANDLE handle = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
                                             THREAD_QUERY_LIMITED_INFORMATION,
                                         FALSE, id);
        if (!handle) {
            continue;
        }
        std::string name = std::to_string(id);
        PWSTR description = nullptr;
        if (SUCCEEDED(GetThreadDescription(handle, &description)) && description && *description) {
            name.clear();
            for (const wchar_t* c = description; *c; ++c) {
                name += *c < 128 ? static_cast<char>(*c) : '?';
            }
        }
        if (description) {
            LocalFree(description);
        }
        CONTEXT ctx{};
        ctx.ContextFlags = CONTEXT_FULL;
        std::array<u64, 64> stack{};
        SIZE_T stack_read = 0;
        const bool suspended = SuspendThread(handle) != DWORD(-1);
        const bool got = suspended && GetThreadContext(handle, &ctx);
        if (got) {
            ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<void*>(ctx.Rsp), stack.data(),
                              sizeof(stack), &stack_read);
        }
        if (suspended) {
            ::ResumeThread(handle);
        }
        CloseHandle(handle);
        if (!got) {
            continue;
        }
        out +=
            fmt::format("\n  thread {} ({}): RIP {:#x} {}", name, id, ctx.Rip, describe(ctx.Rip));
        u32 frames = 0;
        for (size_t i = 0; i < stack_read / sizeof(u64) && frames < 8; ++i) {
            if (stack[i] < 0x10000) {
                continue;
            }
            const auto text = describe(stack[i]);
            if (text.starts_with("host ") || text.find(" Code ") != std::string::npos) {
                out += fmt::format(" | [rsp+{:#x}] {:#x} {}", i * 8, stack[i], text);
                ++frames;
            }
        }
    }
    return out;
#else
    return {};
#endif
}

void DebugStateImpl::AddCurrentThreadToGuestList() {
    std::lock_guard lock{guest_threads_mutex};
    const ThreadID id = ThisThreadID();
    guest_threads.push_back(id);
}

void DebugStateImpl::RemoveCurrentThreadFromGuestList() {
    std::lock_guard lock{guest_threads_mutex};
    const ThreadID id = ThisThreadID();
    std::erase_if(guest_threads, [&](const ThreadID& v) { return v == id; });
}

void DebugStateImpl::PauseGuestThreads() {
    using namespace Libraries::MsgDialog;
    std::unique_lock lock{guest_threads_mutex};
    if (is_guest_threads_paused) {
        return;
    }
    if (ShouldPauseInSubmit()) {
        waiting_submit_pause = false;
        should_show_frame_dump = true;
    }
    bool self_guest = false;
    ThreadID self_id = ThisThreadID();
    for (const auto& id : guest_threads) {
        if (id == self_id) {
            self_guest = true;
        } else {
            PauseThread(id);
        }
    }
    pause_time = Libraries::Kernel::Dev::GetClock()->GetUptime();
    is_guest_threads_paused = true;
    lock.unlock();
    if (self_guest) {
        PauseThread(self_id);
    }
}

void DebugStateImpl::ResumeGuestThreads() {
    std::lock_guard lock{guest_threads_mutex};
    if (!is_guest_threads_paused) {
        return;
    }

    u64 delta_time = Libraries::Kernel::Dev::GetClock()->GetUptime() - pause_time;
    Libraries::Kernel::Dev::GetInitialPtc() += delta_time;
    for (const auto& id : guest_threads) {
        ResumeThread(id);
    }
    is_guest_threads_paused = false;
}

void DebugStateImpl::RequestFrameDump(s32 count) {
    ASSERT(!DumpingCurrentFrame());
    gnm_frame_dump_request_count = count;
    frame_dump_list.clear();
    frame_dump_list.resize(count);
    const auto f = gnm_frame_count.load() + 1;
    for (size_t i = 0; i < count; ++i) {
        frame_dump_list[i].frame_id = f + i;
    }
    waiting_submit_pause = true;
}

void DebugStateImpl::PushQueueDump(QueueDump dump) {
    ASSERT(DumpingCurrentFrame());
    std::unique_lock lock{frame_dump_list_mutex};
    auto& frame = GetFrameDump();
    { // Find draw calls
        auto data = std::span{dump.data};
        auto initial_data = data.data();
        while (!data.empty()) {
            const auto* header = reinterpret_cast<const AmdGpu::PM4Type3Header*>(data.data());
            const auto type = header->type;
            if (type == 2) {
                data = data.subspan(1);
            } else if (type != 3) {
                UNREACHABLE();
            }
            const AmdGpu::PM4ItOpcode opcode = header->opcode;
            if (Core::Devtools::Widget::IsDrawCall(opcode)) {
                const auto offset =
                    reinterpret_cast<uintptr_t>(header) - reinterpret_cast<uintptr_t>(initial_data);
                const auto addr = dump.base_addr + offset;
                waiting_reg_dumps.emplace(addr, &frame);
                waiting_reg_dumps_dbg.emplace(
                    addr,
                    fmt::format("#{} h({}) queue {} {} {}",
                                frame_dump_list.size() - gnm_frame_dump_request_count, addr,
                                magic_enum::enum_name(dump.type), dump.submit_num, dump.num2));
            }
            data = data.subspan(header->NumWords() + 1);
        }
    }
    frame.queues.push_back(std::move(dump));
}

std::optional<RegDump*> DebugStateImpl::GetRegDump(uintptr_t base_addr, uintptr_t header_addr) {
    const auto it = waiting_reg_dumps.find(header_addr);
    if (it == waiting_reg_dumps.end()) {
        return std::nullopt;
    }
    auto& frame = *it->second;
    waiting_reg_dumps.erase(it);
    waiting_reg_dumps_dbg.erase(waiting_reg_dumps_dbg.find(header_addr));
    return &frame.regs[header_addr - base_addr];
}

void DebugStateImpl::PushRegsDump(uintptr_t base_addr, uintptr_t header_addr,
                                  const AmdGpu::Regs& regs) {
    std::scoped_lock lock{frame_dump_list_mutex};

    auto dump = GetRegDump(base_addr, header_addr);
    if (!dump) {
        return;
    }

    (*dump)->regs = regs;

    for (int i = 0; i < RegDump::MaxShaderStages; i++) {
        if ((*dump)->regs.stage_enable.IsStageEnabled(i)) {
            auto stage = (*dump)->regs.ProgramForStage(i);
            if (stage->address) {
                const auto params = AmdGpu::GetParams(*stage);
                (*dump)->stages[i] = PipelineShaderProgramDump{
                    .name = Vulkan::PipelineCache::GetShaderName(Shader::StageFromIndex(i),
                                                                 params.hash),
                    .hash = params.hash,
                    .user_data = *stage,
                    .code = std::vector<u32>{params.code.begin(), params.code.end()},
                };
            }
        }
    }
}

void DebugStateImpl::PushRegsDumpCompute(uintptr_t base_addr, uintptr_t header_addr,
                                         const CsState& cs_state) {
    std::scoped_lock lock{frame_dump_list_mutex};

    auto dump = GetRegDump(base_addr, header_addr);
    if (!dump) {
        return;
    }

    (*dump)->is_compute = true;
    auto& cs = (*dump)->regs.cs_program;
    cs = cs_state;

    const auto params = AmdGpu::GetParams(cs);
    (*dump)->cs_data = PipelineComputerProgramDump{
        .name = Vulkan::PipelineCache::GetShaderName(Shader::HwStage::Compute, params.hash),
        .hash = params.hash,
        .cs_program = cs,
        .code = std::vector<u32>{params.code.begin(), params.code.end()},
    };
}

void DebugStateImpl::CollectShader(const std::string& name, Shader::SwStage l_stage,
                                   vk::ShaderModule module, std::span<const u32> spv,
                                   std::span<const u32> raw_code, std::span<const u32> patch_spv,
                                   bool is_patched) {
    shader_dump_list.emplace_back(name, l_stage, module, std::vector<u32>{spv.begin(), spv.end()},
                                  std::vector<u32>{raw_code.begin(), raw_code.end()},
                                  std::vector<u32>{patch_spv.begin(), patch_spv.end()}, is_patched);
}
