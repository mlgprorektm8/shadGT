// SPDX-FileCopyrightText: Copyright 2026 shadGT
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stop_token>
#include <type_traits>
#include <utility>

#include "common/assert.h"
#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

class Scheduler;

/// PERF-084: which of a session's command buffers a command goes to.
enum class CmdTarget : u8 { Primary, Upload };

/// PERF-084: the command buffer a replayed command goes to (the scheduler's command thread).
vk::CommandBuffer ReplayCommandBuffer(Scheduler& scheduler, CmdTarget target);

/// PERF-084: Vulkan commands recorded on the GPU thread and replayed in order on the scheduler's
/// command thread, which makes the Vulkan calls. In races about 12% of the recorder's time was
/// spent in the driver recording commands; that now runs beside the caches instead of after them.
/// The caches still run on one thread in order, so nothing about what is recorded changes.
/// One thread records at a time (the same rule as for a command buffer); one thread replays.
class CommandRing {
public:
    using ExecFn = void (*)(Scheduler& scheduler, void* fn);

    struct Header {
        ExecFn exec; ///< nullptr: the rest of the ring is unused; continue at its start.
        u32 size;    ///< Bytes to the next command.
        u32 fn_offset;
    };
    static_assert(sizeof(Header) == 16);

    /// Data a command's Vulkan call points at, copied after the command.
    class Extra {
    public:
        Extra(u8* cursor_, u8* limit_) : cursor{cursor_}, limit{limit_} {}

        template <typename T>
        T* Copy(const T* source, size_t count) {
            if (count == 0 || source == nullptr) {
                return nullptr;
            }
            T* dest = Alloc<T>(count);
            std::memcpy(static_cast<void*>(dest), source, count * sizeof(T));
            return dest;
        }

        template <typename T>
        T* Alloc(size_t count) {
            cursor = AlignPtr(cursor, alignof(T));
            T* dest = reinterpret_cast<T*>(cursor);
            cursor += count * sizeof(T);
            ASSERT_MSG(cursor <= limit, "PERF-084: command data larger than reserved");
            return dest;
        }

        u8* Cursor() const noexcept {
            return cursor;
        }

    private:
        u8* cursor;
        u8* limit;
    };

    /// The most bytes Extra::Copy<T> takes for count elements.
    template <typename T>
    static constexpr size_t Bytes(size_t count) noexcept {
        return count * sizeof(T) + alignof(T);
    }

    static constexpr size_t Capacity = size_t{32} << 20;

    CommandRing() : storage{new (std::align_val_t{64}) u8[Capacity]} {}
    ~CommandRing() {
        ::operator delete[](storage, std::align_val_t{64});
    }
    CommandRing(const CommandRing&) = delete;
    CommandRing& operator=(const CommandRing&) = delete;

    /// Records a command for the target command buffer. build(Extra&) copies the data the
    /// command needs and returns a callable taking the vk::CommandBuffer.
    template <CmdTarget target, typename Build>
    void Record(size_t extra_bytes, Build&& build) {
        using F = std::invoke_result_t<Build&, Extra&>;
        Emit<F>(extra_bytes, build, &ExecCommand<F, target>);
    }

    /// Records a function the command thread runs with the scheduler (sessions, submits).
    template <typename F>
    void RecordControl(F&& fn) {
        using G = std::decay_t<F>;
        auto build = [&](Extra&) -> G { return G(std::forward<F>(fn)); };
        Emit<G>(0, build, &ExecControl<G>);
    }

    /// Bytes recorded so far (a position the command thread reaches).
    u64 Recorded() const noexcept {
        return head_local;
    }

    /// Bytes the command thread replayed so far.
    u64 Replayed() const noexcept {
        return tail.load(std::memory_order_acquire);
    }

    /// Command thread: the published end.
    u64 Published() const noexcept {
        return head.load(std::memory_order_acquire);
    }

    /// Command thread: the command at a position.
    Header* At(u64 position) const noexcept {
        return reinterpret_cast<Header*>(storage + (position & (Capacity - 1)));
    }

    /// Command thread: reports progress.
    void SetReplayed(u64 position) noexcept {
        tail.store(position, std::memory_order_release);
    }

    /// Command thread: waits for commands past `seen` (or a wake).
    void WaitForWork(u64 seen, const std::stop_token& stoken);

    /// Wakes the command thread (to stop it).
    void Wake() noexcept {
        sleeping.store(false, std::memory_order_relaxed);
        wake.fetch_add(1, std::memory_order_release);
        wake.notify_one();
    }

    u64 full_waits{}; ///< Times the recorder waited for space.

private:
    template <typename T>
    static T* AlignPtr(T* ptr, size_t align) noexcept {
        const auto value = reinterpret_cast<uintptr_t>(ptr);
        return reinterpret_cast<T*>((value + align - 1) & ~(uintptr_t(align) - 1));
    }

    template <typename F, CmdTarget target>
    static void ExecCommand(Scheduler& scheduler, void* fn) {
        F& f = *static_cast<F*>(fn);
        f(ReplayCommandBuffer(scheduler, target));
        f.~F();
    }

    template <typename F>
    static void ExecControl(Scheduler& scheduler, void* fn) {
        F& f = *static_cast<F*>(fn);
        f(scheduler);
        f.~F();
    }

    /// Layout: [Header][F][extra data]. F is built in place, after which it copies its data.
    template <typename F, typename Build>
    void Emit(size_t extra_bytes, Build& build, ExecFn exec) {
        static_assert(alignof(F) <= 16);
        const size_t reserve = sizeof(Header) + sizeof(F) + extra_bytes + 32;
        u8* const base = Reserve(reserve);
        u8* const fn_at = base + sizeof(Header);
        Extra extra{fn_at + sizeof(F), base + reserve};
        new (fn_at) F(build(extra));
        auto* header = reinterpret_cast<Header*>(base);
        header->exec = exec;
        header->fn_offset = static_cast<u32>(fn_at - base);
        header->size = static_cast<u32>(AlignPtr(extra.Cursor(), 16) - base);
        Publish(header->size);
    }

    u8* Reserve(size_t bytes) {
        ASSERT_MSG(bytes <= Capacity / 4, "PERF-084: command of {} bytes", bytes);
        size_t offset = head_local & (Capacity - 1);
        const bool wrap = offset + bytes > Capacity;
        const size_t needed = wrap ? bytes + (Capacity - offset) : bytes;
        if (head_local + needed - tail.load(std::memory_order_acquire) > Capacity) {
            WaitForSpace(needed);
        }
        if (wrap) {
            auto* header = reinterpret_cast<Header*>(storage + offset);
            header->exec = nullptr;
            header->size = static_cast<u32>(Capacity - offset);
            header->fn_offset = 0;
            head_local += Capacity - offset;
            offset = 0;
        }
        return storage + offset;
    }

    void WaitForSpace(size_t needed);

    void Publish(u32 bytes) noexcept {
        head_local += bytes;
        head.store(head_local, std::memory_order_seq_cst);
        if (sleeping.load(std::memory_order_seq_cst)) {
            Wake();
        }
    }

    u8* storage;
    u64 head_local{}; ///< Recorder side.
    alignas(64) std::atomic<u64> head{};
    alignas(64) std::atomic<u64> tail{};
    alignas(64) std::atomic<u32> wake{};
    std::atomic<bool> sleeping{};
};

/// PERF-084: what Scheduler::CommandBuffer() returns. Records into the scheduler's command ring
/// when it has a command thread, else calls the command buffer directly. The methods are the
/// vk::CommandBuffer calls the renderer makes, with the same arguments; data they point at is
/// copied when recorded.
class CmdBuf {
public:
    explicit CmdBuf(vk::CommandBuffer direct_) noexcept : direct{direct_} {}
    CmdBuf(CommandRing* ring_, CmdTarget target_) noexcept : ring{ring_}, target{target_} {}

    /// The command buffer of a direct handle (nullptr handle when recording into a ring).
    vk::CommandBuffer Direct() const noexcept {
        return direct;
    }

    void bindPipeline(vk::PipelineBindPoint bind_point, vk::Pipeline pipeline) const {
        Call([=](vk::CommandBuffer cmd) { cmd.bindPipeline(bind_point, pipeline); });
    }

    void draw(u32 vertex_count, u32 instance_count, u32 first_vertex, u32 first_instance) const {
        Call([=](vk::CommandBuffer cmd) {
            cmd.draw(vertex_count, instance_count, first_vertex, first_instance);
        });
    }

    void drawIndexed(u32 index_count, u32 instance_count, u32 first_index, s32 vertex_offset,
                     u32 first_instance) const {
        Call([=](vk::CommandBuffer cmd) {
            cmd.drawIndexed(index_count, instance_count, first_index, vertex_offset,
                            first_instance);
        });
    }

    void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, u32 draw_count,
                      u32 stride) const {
        Call([=](vk::CommandBuffer cmd) { cmd.drawIndirect(buffer, offset, draw_count, stride); });
    }

    void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, u32 draw_count,
                             u32 stride) const {
        Call([=](vk::CommandBuffer cmd) {
            cmd.drawIndexedIndirect(buffer, offset, draw_count, stride);
        });
    }

    void drawIndirectCount(vk::Buffer buffer, vk::DeviceSize offset, vk::Buffer count_buffer,
                           vk::DeviceSize count_offset, u32 max_draw_count, u32 stride) const {
        Call([=](vk::CommandBuffer cmd) {
            cmd.drawIndirectCount(buffer, offset, count_buffer, count_offset, max_draw_count,
                                  stride);
        });
    }

    void drawIndexedIndirectCount(vk::Buffer buffer, vk::DeviceSize offset,
                                  vk::Buffer count_buffer, vk::DeviceSize count_offset,
                                  u32 max_draw_count, u32 stride) const {
        Call([=](vk::CommandBuffer cmd) {
            cmd.drawIndexedIndirectCount(buffer, offset, count_buffer, count_offset,
                                         max_draw_count, stride);
        });
    }

    void dispatch(u32 x, u32 y, u32 z) const {
        Call([=](vk::CommandBuffer cmd) { cmd.dispatch(x, y, z); });
    }

    /// NAT-100: runs a recorded secondary command buffer here.
    void executeCommands(vk::CommandBuffer secondary) const {
        Call([=](vk::CommandBuffer cmd) { cmd.executeCommands(secondary); });
    }

    void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
        Call([=](vk::CommandBuffer cmd) { cmd.dispatchIndirect(buffer, offset); });
    }

    void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
        Call([=](vk::CommandBuffer cmd) { cmd.bindIndexBuffer(buffer, offset, type); });
    }

    void bindVertexBuffers(u32 first_binding, u32 binding_count, const vk::Buffer* buffers,
                           const vk::DeviceSize* offsets) const {
        if (!ring) {
            direct.bindVertexBuffers(first_binding, binding_count, buffers, offsets);
            return;
        }
        Record(CommandRing::Bytes<vk::Buffer>(binding_count) +
                   CommandRing::Bytes<vk::DeviceSize>(binding_count),
               [&](CommandRing::Extra& extra) {
                   const auto* b = extra.Copy(buffers, binding_count);
                   const auto* o = extra.Copy(offsets, binding_count);
                   return [=](vk::CommandBuffer cmd) {
                       cmd.bindVertexBuffers(first_binding, binding_count, b, o);
                   };
               });
    }

    void bindVertexBuffers2(u32 first_binding, u32 binding_count, const vk::Buffer* buffers,
                            const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
                            const vk::DeviceSize* strides) const {
        if (!ring) {
            direct.bindVertexBuffers2(first_binding, binding_count, buffers, offsets, sizes,
                                      strides);
            return;
        }
        Record(CommandRing::Bytes<vk::Buffer>(binding_count) +
                   3 * CommandRing::Bytes<vk::DeviceSize>(binding_count),
               [&](CommandRing::Extra& extra) {
                   const auto* b = extra.Copy(buffers, binding_count);
                   const auto* o = extra.Copy(offsets, binding_count);
                   const auto* sz = extra.Copy(sizes, binding_count);
                   const auto* st = extra.Copy(strides, binding_count);
                   return [=](vk::CommandBuffer cmd) {
                       cmd.bindVertexBuffers2(first_binding, binding_count, b, o, sz, st);
                   };
               });
    }

    void setVertexInputEXT(
        vk::ArrayProxy<const vk::VertexInputBindingDescription2EXT> const& bindings,
        vk::ArrayProxy<const vk::VertexInputAttributeDescription2EXT> const& attributes) const {
        if (!ring) {
            direct.setVertexInputEXT(bindings, attributes);
            return;
        }
        const u32 num_bindings = bindings.size();
        const u32 num_attributes = attributes.size();
        Record(CommandRing::Bytes<vk::VertexInputBindingDescription2EXT>(num_bindings) +
                   CommandRing::Bytes<vk::VertexInputAttributeDescription2EXT>(num_attributes),
               [&](CommandRing::Extra& extra) {
                   const auto* b = extra.Copy(bindings.data(), num_bindings);
                   const auto* a = extra.Copy(attributes.data(), num_attributes);
                   return [=](vk::CommandBuffer cmd) {
                       cmd.setVertexInputEXT(num_bindings, b, num_attributes, a);
                   };
               });
    }

    void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, u32 offset,
                       u32 size, const void* values) const {
        if (!ring) {
            direct.pushConstants(layout, stages, offset, size, values);
            return;
        }
        Record(CommandRing::Bytes<u8>(size) + 16, [&](CommandRing::Extra& extra) {
            const auto* data = extra.Copy(static_cast<const u8*>(values), size);
            return [=](vk::CommandBuffer cmd) {
                cmd.pushConstants(layout, stages, offset, size, data);
            };
        });
    }

    void pushDescriptorSetKHR(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout, u32 set,
                              vk::ArrayProxy<const vk::WriteDescriptorSet> const& writes) const {
        if (!ring) {
            direct.pushDescriptorSetKHR(bind_point, layout, set, writes);
            return;
        }
        Record(DescriptorWriteBytes(writes), [&](CommandRing::Extra& extra) {
            const auto* copied = CopyDescriptorWrites(extra, writes);
            const u32 count = writes.size();
            return [=](vk::CommandBuffer cmd) {
                cmd.pushDescriptorSetKHR(bind_point, layout, set, count, copied);
            };
        });
    }

    void bindDescriptorSets(vk::PipelineBindPoint bind_point, vk::PipelineLayout layout,
                            u32 first_set, vk::ArrayProxy<const vk::DescriptorSet> const& sets,
                            vk::ArrayProxy<const u32> const& dynamic_offsets) const {
        if (!ring) {
            direct.bindDescriptorSets(bind_point, layout, first_set, sets, dynamic_offsets);
            return;
        }
        const u32 num_sets = sets.size();
        const u32 num_offsets = dynamic_offsets.size();
        Record(CommandRing::Bytes<vk::DescriptorSet>(num_sets) +
                   CommandRing::Bytes<u32>(num_offsets),
               [&](CommandRing::Extra& extra) {
                   const auto* s = extra.Copy(sets.data(), num_sets);
                   const auto* o = extra.Copy(dynamic_offsets.data(), num_offsets);
                   return [=](vk::CommandBuffer cmd) {
                       cmd.bindDescriptorSets(bind_point, layout, first_set, num_sets, s,
                                              num_offsets, o);
                   };
               });
    }

    void pipelineBarrier2(const vk::DependencyInfo& info) const {
        if (!ring) {
            direct.pipelineBarrier2(info);
            return;
        }
        ASSERT(info.pNext == nullptr);
        Record(CommandRing::Bytes<vk::MemoryBarrier2>(info.memoryBarrierCount) +
                   CommandRing::Bytes<vk::BufferMemoryBarrier2>(info.bufferMemoryBarrierCount) +
                   CommandRing::Bytes<vk::ImageMemoryBarrier2>(info.imageMemoryBarrierCount),
               [&](CommandRing::Extra& extra) {
                   vk::DependencyInfo copy = info;
                   copy.pMemoryBarriers =
                       extra.Copy(info.pMemoryBarriers, info.memoryBarrierCount);
                   copy.pBufferMemoryBarriers =
                       extra.Copy(info.pBufferMemoryBarriers, info.bufferMemoryBarrierCount);
                   copy.pImageMemoryBarriers =
                       extra.Copy(info.pImageMemoryBarriers, info.imageMemoryBarrierCount);
                   return [copy](vk::CommandBuffer cmd) { cmd.pipelineBarrier2(copy); };
               });
    }

    void beginRendering(const vk::RenderingInfo& info) const {
        if (!ring) {
            direct.beginRendering(info);
            return;
        }
        ASSERT(info.pNext == nullptr);
        Record(CommandRing::Bytes<vk::RenderingAttachmentInfo>(info.colorAttachmentCount + 2),
               [&](CommandRing::Extra& extra) {
                   vk::RenderingInfo copy = info;
                   copy.pColorAttachments =
                       extra.Copy(info.pColorAttachments, info.colorAttachmentCount);
                   copy.pDepthAttachment = extra.Copy(info.pDepthAttachment, 1);
                   copy.pStencilAttachment = extra.Copy(info.pStencilAttachment, 1);
                   return [copy](vk::CommandBuffer cmd) { cmd.beginRendering(copy); };
               });
    }

    void endRendering() const {
        Call([](vk::CommandBuffer cmd) { cmd.endRendering(); });
    }

    void copyBuffer(vk::Buffer src, vk::Buffer dst,
                    vk::ArrayProxy<const vk::BufferCopy> const& regions) const {
        if (!ring) {
            direct.copyBuffer(src, dst, regions);
            return;
        }
        RecordRegions(regions, [=](vk::CommandBuffer cmd, u32 count, const vk::BufferCopy* r) {
            cmd.copyBuffer(src, dst, count, r);
        });
    }

    void copyBufferToImage(vk::Buffer src, vk::Image dst, vk::ImageLayout layout,
                           vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
        if (!ring) {
            direct.copyBufferToImage(src, dst, layout, regions);
            return;
        }
        RecordRegions(regions,
                      [=](vk::CommandBuffer cmd, u32 count, const vk::BufferImageCopy* r) {
                          cmd.copyBufferToImage(src, dst, layout, count, r);
                      });
    }

    void copyImageToBuffer(vk::Image src, vk::ImageLayout layout, vk::Buffer dst,
                           vk::ArrayProxy<const vk::BufferImageCopy> const& regions) const {
        if (!ring) {
            direct.copyImageToBuffer(src, layout, dst, regions);
            return;
        }
        RecordRegions(regions,
                      [=](vk::CommandBuffer cmd, u32 count, const vk::BufferImageCopy* r) {
                          cmd.copyImageToBuffer(src, layout, dst, count, r);
                      });
    }

    void copyImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
                   vk::ImageLayout dst_layout,
                   vk::ArrayProxy<const vk::ImageCopy> const& regions) const {
        if (!ring) {
            direct.copyImage(src, src_layout, dst, dst_layout, regions);
            return;
        }
        RecordRegions(regions, [=](vk::CommandBuffer cmd, u32 count, const vk::ImageCopy* r) {
            cmd.copyImage(src, src_layout, dst, dst_layout, count, r);
        });
    }

    void resolveImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst,
                      vk::ImageLayout dst_layout,
                      vk::ArrayProxy<const vk::ImageResolve> const& regions) const {
        if (!ring) {
            direct.resolveImage(src, src_layout, dst, dst_layout, regions);
            return;
        }
        RecordRegions(regions, [=](vk::CommandBuffer cmd, u32 count, const vk::ImageResolve* r) {
            cmd.resolveImage(src, src_layout, dst, dst_layout, count, r);
        });
    }

    void clearColorImage(vk::Image image, vk::ImageLayout layout,
                         const vk::ClearColorValue& color,
                         vk::ArrayProxy<const vk::ImageSubresourceRange> const& ranges) const {
        if (!ring) {
            direct.clearColorImage(image, layout, color, ranges);
            return;
        }
        RecordRegions(ranges,
                      [=](vk::CommandBuffer cmd, u32 count, const vk::ImageSubresourceRange* r) {
                          cmd.clearColorImage(image, layout, &color, count, r);
                      });
    }

    void fillBuffer(vk::Buffer dst, vk::DeviceSize offset, vk::DeviceSize size, u32 data) const {
        Call([=](vk::CommandBuffer cmd) { cmd.fillBuffer(dst, offset, size, data); });
    }

    void updateBuffer(vk::Buffer dst, vk::DeviceSize offset, vk::DeviceSize size,
                      const void* data) const {
        if (!ring) {
            direct.updateBuffer(dst, offset, size, data);
            return;
        }
        Record(CommandRing::Bytes<u8>(size) + 16, [&](CommandRing::Extra& extra) {
            const auto* copied = extra.Copy(static_cast<const u8*>(data), size);
            return [=](vk::CommandBuffer cmd) { cmd.updateBuffer(dst, offset, size, copied); };
        });
    }

    void setViewportWithCount(vk::ArrayProxy<const vk::Viewport> const& viewports) const {
        if (!ring) {
            direct.setViewportWithCount(viewports);
            return;
        }
        RecordRegions(viewports, [](vk::CommandBuffer cmd, u32 count, const vk::Viewport* v) {
            cmd.setViewportWithCount(count, v);
        });
    }

    void setScissorWithCount(vk::ArrayProxy<const vk::Rect2D> const& scissors) const {
        if (!ring) {
            direct.setScissorWithCount(scissors);
            return;
        }
        RecordRegions(scissors, [](vk::CommandBuffer cmd, u32 count, const vk::Rect2D* s) {
            cmd.setScissorWithCount(count, s);
        });
    }

    void setColorWriteMaskEXT(u32 first_attachment,
                              vk::ArrayProxy<const vk::ColorComponentFlags> const& masks) const {
        if (!ring) {
            direct.setColorWriteMaskEXT(first_attachment, masks);
            return;
        }
        RecordRegions(masks,
                      [=](vk::CommandBuffer cmd, u32 count, const vk::ColorComponentFlags* m) {
                          cmd.setColorWriteMaskEXT(first_attachment, count, m);
                      });
    }

    void setBlendConstants(const float constants[4]) const {
        const std::array<float, 4> values{constants[0], constants[1], constants[2], constants[3]};
        Call([=](vk::CommandBuffer cmd) { cmd.setBlendConstants(values.data()); });
    }

    void setDepthTestEnable(bool enable) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setDepthTestEnable(enable); });
    }
    void setDepthWriteEnable(bool enable) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setDepthWriteEnable(enable); });
    }
    void setDepthCompareOp(vk::CompareOp op) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setDepthCompareOp(op); });
    }
    void setDepthBoundsTestEnable(bool enable) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setDepthBoundsTestEnable(enable); });
    }
    void setDepthBounds(float min, float max) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setDepthBounds(min, max); });
    }
    void setDepthBiasEnable(bool enable) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setDepthBiasEnable(enable); });
    }
    void setDepthBias(float constant, float clamp, float slope) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setDepthBias(constant, clamp, slope); });
    }
    void setStencilTestEnable(bool enable) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setStencilTestEnable(enable); });
    }
    void setStencilOp(vk::StencilFaceFlags faces, vk::StencilOp fail, vk::StencilOp pass,
                      vk::StencilOp depth_fail, vk::CompareOp compare) const {
        Call([=](vk::CommandBuffer cmd) {
            cmd.setStencilOp(faces, fail, pass, depth_fail, compare);
        });
    }
    void setStencilReference(vk::StencilFaceFlags faces, u32 reference) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setStencilReference(faces, reference); });
    }
    void setStencilWriteMask(vk::StencilFaceFlags faces, u32 mask) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setStencilWriteMask(faces, mask); });
    }
    void setStencilCompareMask(vk::StencilFaceFlags faces, u32 mask) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setStencilCompareMask(faces, mask); });
    }
    void setPrimitiveRestartEnable(bool enable) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setPrimitiveRestartEnable(enable); });
    }
    void setRasterizerDiscardEnable(bool enable) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setRasterizerDiscardEnable(enable); });
    }
    void setCullMode(vk::CullModeFlags mode) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setCullMode(mode); });
    }
    void setFrontFace(vk::FrontFace face) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setFrontFace(face); });
    }
    void setLineWidth(float width) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setLineWidth(width); });
    }
    void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags aspects) const {
        Call([=](vk::CommandBuffer cmd) { cmd.setAttachmentFeedbackLoopEnableEXT(aspects); });
    }

    void beginDebugUtilsLabelEXT(const vk::DebugUtilsLabelEXT& label) const {
        if (!ring) {
            direct.beginDebugUtilsLabelEXT(label);
            return;
        }
        RecordLabel(label, false);
    }
    void insertDebugUtilsLabelEXT(const vk::DebugUtilsLabelEXT& label) const {
        if (!ring) {
            direct.insertDebugUtilsLabelEXT(label);
            return;
        }
        RecordLabel(label, true);
    }
    void endDebugUtilsLabelEXT() const {
        Call([](vk::CommandBuffer cmd) { cmd.endDebugUtilsLabelEXT(); });
    }

private:
    /// A command without data to copy.
    template <typename F>
    void Call(F&& fn) const {
        if (!ring) {
            fn(direct);
            return;
        }
        Record(0, [&](CommandRing::Extra&) { return std::forward<F>(fn); });
    }

    template <typename Build>
    void Record(size_t extra_bytes, Build&& build) const {
        if (target == CmdTarget::Primary) {
            ring->Record<CmdTarget::Primary>(extra_bytes, build);
        } else {
            ring->Record<CmdTarget::Upload>(extra_bytes, build);
        }
    }

    template <typename T, typename F>
    void RecordRegions(vk::ArrayProxy<const T> const& regions, F fn) const {
        const u32 count = regions.size();
        Record(CommandRing::Bytes<T>(count), [&](CommandRing::Extra& extra) {
            const T* copied = extra.Copy(regions.data(), count);
            return [=](vk::CommandBuffer cmd) { fn(cmd, count, copied); };
        });
    }

    void RecordLabel(const vk::DebugUtilsLabelEXT& label, bool insert) const {
        const size_t length = label.pLabelName ? std::strlen(label.pLabelName) : 0;
        Record(length + 1, [&](CommandRing::Extra& extra) {
            char* name = extra.Alloc<char>(length + 1);
            if (length != 0) {
                std::memcpy(name, label.pLabelName, length);
            }
            name[length] = '\0';
            vk::DebugUtilsLabelEXT copy = label;
            copy.pLabelName = name;
            return [copy, insert](vk::CommandBuffer cmd) {
                if (insert) {
                    cmd.insertDebugUtilsLabelEXT(copy);
                } else {
                    cmd.beginDebugUtilsLabelEXT(copy);
                }
            };
        });
    }

    static bool UsesImageInfo(vk::DescriptorType type) noexcept {
        switch (type) {
        case vk::DescriptorType::eSampler:
        case vk::DescriptorType::eCombinedImageSampler:
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eInputAttachment:
            return true;
        default:
            return false;
        }
    }

    static bool UsesTexelBuffer(vk::DescriptorType type) noexcept {
        return type == vk::DescriptorType::eUniformTexelBuffer ||
               type == vk::DescriptorType::eStorageTexelBuffer;
    }

    static size_t DescriptorWriteBytes(vk::ArrayProxy<const vk::WriteDescriptorSet> const& writes) {
        size_t bytes = CommandRing::Bytes<vk::WriteDescriptorSet>(writes.size());
        for (const auto& write : writes) {
            if (UsesImageInfo(write.descriptorType)) {
                bytes += CommandRing::Bytes<vk::DescriptorImageInfo>(write.descriptorCount);
            } else if (UsesTexelBuffer(write.descriptorType)) {
                bytes += CommandRing::Bytes<vk::BufferView>(write.descriptorCount);
            } else {
                bytes += CommandRing::Bytes<vk::DescriptorBufferInfo>(write.descriptorCount);
            }
        }
        return bytes;
    }

    static const vk::WriteDescriptorSet* CopyDescriptorWrites(
        CommandRing::Extra& extra, vk::ArrayProxy<const vk::WriteDescriptorSet> const& writes) {
        auto* copied = extra.Copy(writes.data(), writes.size());
        for (u32 i = 0; i < writes.size(); ++i) {
            auto& write = copied[i];
            ASSERT(write.pNext == nullptr);
            if (UsesImageInfo(write.descriptorType)) {
                write.pImageInfo = extra.Copy(write.pImageInfo, write.descriptorCount);
                write.pBufferInfo = nullptr;
                write.pTexelBufferView = nullptr;
            } else if (UsesTexelBuffer(write.descriptorType)) {
                write.pTexelBufferView = extra.Copy(write.pTexelBufferView, write.descriptorCount);
                write.pImageInfo = nullptr;
                write.pBufferInfo = nullptr;
            } else {
                write.pBufferInfo = extra.Copy(write.pBufferInfo, write.descriptorCount);
                write.pImageInfo = nullptr;
                write.pTexelBufferView = nullptr;
            }
        }
        return copied;
    }

    vk::CommandBuffer direct{};
    CommandRing* ring{};
    CmdTarget target{CmdTarget::Primary};
};

} // namespace Vulkan
