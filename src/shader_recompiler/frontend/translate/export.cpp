// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/frontend/translate/translate.h"
#include "shader_recompiler/ir/position.h"
#include "shader_recompiler/ir/reinterpret.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Gcn {

static AmdGpu::NumberFormat NumberFormatCompressed(AmdGpu::ShaderExportFormat export_format) {
    switch (export_format) {
    case AmdGpu::ShaderExportFormat::ABGR_FP16:
        return AmdGpu::NumberFormat::Float;
    case AmdGpu::ShaderExportFormat::ABGR_UNORM16:
        return AmdGpu::NumberFormat::Unorm;
    case AmdGpu::ShaderExportFormat::ABGR_SNORM16:
        return AmdGpu::NumberFormat::Snorm;
    case AmdGpu::ShaderExportFormat::ABGR_UINT16:
        return AmdGpu::NumberFormat::Uint;
    case AmdGpu::ShaderExportFormat::ABGR_SINT16:
        return AmdGpu::NumberFormat::Sint;
    default:
        UNREACHABLE_MSG("Unimplemented compressed export format {}",
                        static_cast<u32>(export_format));
    }
}

static u32 MaskFromExportFormat(u8 mask, AmdGpu::ShaderExportFormat export_format) {
    switch (export_format) {
    case AmdGpu::ShaderExportFormat::R_32:
        // Red only
        return mask & 1;
    case AmdGpu::ShaderExportFormat::GR_32:
        // Red and Green only
        return mask & 3;
    case AmdGpu::ShaderExportFormat::AR_32:
        // Red and Alpha only
        return mask & 9;
    case AmdGpu::ShaderExportFormat::ABGR_32:
        // All components
        return mask;
    default:
        UNREACHABLE_MSG("Unimplemented uncompressed export format {}",
                        static_cast<u32>(export_format));
    }
}

void Translator::ExportRenderTarget(const GcnInst& inst) {
    const auto& exp = inst.control.exp;
    const IR::Attribute mrt{exp.target};
    const auto& cb0 = runtime_info.hw.fs.color_buffers[0];
    if ((cb0.blend_swizzled_alpha || cb0.blend_swizzled_factors) &&
        mrt != IR::Attribute::RenderTarget0) {
        // Swizzled blend emulation requires a Zero export format on every other MRT, so hardware
        // discards these exports. Keep them from becoming the synthetic second source.
        return;
    }
    info.mrt_mask |= 1u << static_cast<u8>(mrt);

    // Dual source blending uses MRT1 for exporting src1
    u32 color_buffer_idx = static_cast<u32>(mrt) - static_cast<u32>(IR::Attribute::RenderTarget0);
    if (runtime_info.hw.fs.dual_source_blending && mrt == IR::Attribute::RenderTarget1) {
        color_buffer_idx = 0;
    }

    const auto color_buffer = runtime_info.hw.fs.color_buffers[color_buffer_idx];
    if (color_buffer.export_format == AmdGpu::ShaderExportFormat::Zero || exp.en == 0) {
        // No export
        return;
    }

    std::array<IR::F32, 4> components{};
    if (exp.compr) {
        // Components are float16 packed into a VGPR
        const auto num_format = NumberFormatCompressed(color_buffer.export_format);
        // Export R, G
        if (exp.en & 1) {
            const IR::Value unpacked_value =
                ir.Unpack2x16(num_format, ir.GetVectorReg(IR::VectorReg(inst.src[0].code)));
            components[0] = IR::F32{ir.CompositeExtract(unpacked_value, 0)};
            components[1] = IR::F32{ir.CompositeExtract(unpacked_value, 1)};
        }
        // Export B, A
        if ((exp.en >> 2) & 1) {
            const IR::Value unpacked_value =
                ir.Unpack2x16(num_format, ir.GetVectorReg(IR::VectorReg(inst.src[1].code)));
            components[2] = IR::F32{ir.CompositeExtract(unpacked_value, 0)};
            components[3] = IR::F32{ir.CompositeExtract(unpacked_value, 1)};
        }
    } else {
        // Components are float32 into separate VGPRS
        u32 mask = MaskFromExportFormat(exp.en, color_buffer.export_format);
        for (u32 i = 0; i < 4; i++, mask >>= 1) {
            if ((mask & 1) == 0) {
                continue;
            }
            components[i] = ir.GetVectorReg<IR::F32>(IR::VectorReg(inst.src[i].code));
        }
    }

    // Metal seems to have an issue where 8-bit unorm/snorm/sRGB outputs to render target
    // need a bias applied to round correctly; detect and set the flag for that here.
    const auto needs_unorm_fixup = profile.needs_unorm_fixup &&
                                   (color_buffer.num_format == AmdGpu::NumberFormat::Unorm ||
                                    color_buffer.num_format == AmdGpu::NumberFormat::Snorm ||
                                    color_buffer.num_format == AmdGpu::NumberFormat::Srgb) &&
                                   (color_buffer.data_format == AmdGpu::DataFormat::Format8 ||
                                    color_buffer.data_format == AmdGpu::DataFormat::Format8_8 ||
                                    color_buffer.data_format == AmdGpu::DataFormat::Format8_8_8_8);

    if (color_buffer.blend_swizzled_alpha && mrt == IR::Attribute::RenderTarget0) {
        const auto alpha = components[3].IsEmpty() ? ir.Imm32(1.f) : components[3];
        for (u32 i = 0; i < 4; ++i) {
            const auto factor = color_buffer.swizzle.Map(i) == 3 ? ir.Imm32(0.f) : alpha;
            ir.SetAttribute(IR::Attribute::RenderTarget1, factor, i);
        }
    }

    if (color_buffer.blend_swizzled_factors && mrt == IR::Attribute::RenderTarget0) {
        using Factor = AmdGpu::BlendControl::BlendFactor;
        // Fixed-function blending clamps normalized sources and factors to the format range.
        const auto clamp_source = [&](const IR::F32& value) -> IR::F32 {
            switch (color_buffer.num_format) {
            case AmdGpu::NumberFormat::Unorm:
            case AmdGpu::NumberFormat::Srgb:
                return IR::F32{ir.FPClamp(value, ir.Imm32(0.f), ir.Imm32(1.f))};
            case AmdGpu::NumberFormat::Snorm:
                return IR::F32{ir.FPClamp(value, ir.Imm32(-1.f), ir.Imm32(1.f))};
            default:
                return value;
            }
        };
        std::array<IR::F32, 4> source;
        for (u32 i = 0; i < 4; ++i) {
            // An alpha the shader does not export blends as one, matching the exact path.
            source[i] = components[i].IsEmpty() ? ir.Imm32(i == 3 ? 1.f : 0.f)
                                                : clamp_source(components[i]);
        }
        // Each factor applies to one logical channel; SrcColor on alpha reads source alpha.
        const auto factor = [&](Factor blend_factor, u32 channel) -> IR::F32 {
            switch (blend_factor) {
            case Factor::Zero:
                return ir.Imm32(0.f);
            case Factor::One:
                return ir.Imm32(1.f);
            case Factor::SrcColor:
                return source[channel];
            case Factor::OneMinusSrcColor:
                return IR::F32{ir.FPSub(ir.Imm32(1.f), source[channel])};
            case Factor::SrcAlpha:
                return source[3];
            case Factor::OneMinusSrcAlpha:
                return IR::F32{ir.FPSub(ir.Imm32(1.f), source[3])};
            default:
                UNREACHABLE_MSG("Unexpected swizzled blend factor {}", u32(blend_factor));
            }
        };
        // Secondary source in physical lane order: one minus that lane's destination factor.
        for (u32 i = 0; i < 4; ++i) {
            const u32 channel = color_buffer.swizzle.Map(i);
            const auto dst_factor =
                channel == 3 ? color_buffer.swizzled_alpha_dst : color_buffer.swizzled_color_dst;
            const auto one_minus = dst_factor == Factor::Zero  ? ir.Imm32(1.f)
                                   : dst_factor == Factor::One ? ir.Imm32(0.f)
                                                               : IR::F32{ir.FPSub(
                                                                     ir.Imm32(1.f),
                                                                     factor(dst_factor, channel))};
            ir.SetAttribute(IR::Attribute::RenderTarget1, one_minus, i);
        }
        // Primary source premultiplied by each logical channel's source factor.
        for (u32 channel = 0; channel < 4; ++channel) {
            if (components[channel].IsEmpty()) {
                continue;
            }
            const auto src_factor =
                channel == 3 ? color_buffer.swizzled_alpha_src : color_buffer.swizzled_color_src;
            components[channel] =
                src_factor == Factor::Zero  ? ir.Imm32(0.f)
                : src_factor == Factor::One ? source[channel]
                                            : IR::F32{ir.FPMul(factor(src_factor, channel),
                                                               source[channel])};
        }
    }

    // Swizzle components and export
    for (u32 i = 0; i < 4; ++i) {
        const auto swizzled_comp = components[color_buffer.swizzle.Map(i)];
        if (swizzled_comp.IsEmpty()) {
            continue;
        }
        auto converted = ApplyWriteNumberConversion(ir, swizzled_comp, color_buffer.num_conversion);
        if (needs_unorm_fixup) {
            // FIXME: Fix-up for GPUs where float-to-unorm rounding is off from expected.
            converted = ir.FPSub(converted, ir.Imm32(1.f / 127500.f));
        }
        ir.SetAttribute(mrt, converted, i);
    }
}

void Translator::ExportDepth(const GcnInst& inst) {
    const auto& exp = inst.control.exp;
    if (exp.en == 0) {
        // No export
        return;
    }

    std::array<IR::F32, 4> components{};
    if (exp.compr) {
        // Components are float16 packed into a VGPR
        const auto num_format = NumberFormatCompressed(runtime_info.hw.fs.z_export_format);
        // Export R, G
        if (exp.en & 1) {
            const IR::Value unpacked_value =
                ir.Unpack2x16(num_format, ir.GetVectorReg(IR::VectorReg(inst.src[0].code)));
            components[0] = IR::F32{ir.CompositeExtract(unpacked_value, 0)};
            components[1] = IR::F32{ir.CompositeExtract(unpacked_value, 1)};
        }
        // Export B, A
        if ((exp.en >> 2) & 1) {
            const IR::Value unpacked_value =
                ir.Unpack2x16(num_format, ir.GetVectorReg(IR::VectorReg(inst.src[1].code)));
            components[2] = IR::F32{ir.CompositeExtract(unpacked_value, 0)};
            // components[3] = IR::F32{ir.CompositeExtract(unpacked_value, 1)};
        }
    } else {
        // Components are float32 into separate VGPRS
        u32 mask = MaskFromExportFormat(exp.en & runtime_info.hw.fs.mrtz_mask,
                                        runtime_info.hw.fs.z_export_format);
        for (u32 i = 0; i < 4; i++, mask >>= 1) {
            if ((mask & 1) == 0) {
                continue;
            }
            components[i] = ir.GetVectorReg<IR::F32>(IR::VectorReg(inst.src[i].code));
        }
    }

    static constexpr std::array MrtzBuiltins = {IR::Attribute::Depth, IR::Attribute::StencilRef,
                                                IR::Attribute::SampleMask, IR::Attribute::Null};
    for (u32 i = 0; i < 4; ++i) {
        if (components[i].IsEmpty()) {
            continue;
        }
        ir.SetAttribute(MrtzBuiltins[i], components[i]);
    }
}

void Translator::EmitExport(const GcnInst& inst) {
    if (info.hw_stage == HwStage::Fragment && inst.control.exp.vm) {
        ir.Discard(ir.LogicalNot(ir.GetExec()));
    }

    const IR::Attribute attrib{inst.control.exp.target};
    if (IR::IsMrt(attrib)) {
        return ExportRenderTarget(inst);
    }
    if (attrib == IR::Attribute::Depth) {
        return ExportDepth(inst);
    }

    ASSERT_MSG(!inst.control.exp.compr, "Compressed exports only supported for render targets");

    const bool tess_emulated_primitive =
        info.sw_stage == SwStage::Vertex && runtime_info.sw.vs.tess_emulated_primitive;

    u32 mask = inst.control.exp.en;
    for (u32 i = 0; i < 4; i++, mask >>= 1) {
        if ((mask & 1) == 0) {
            continue;
        }
        const auto value = ir.GetVectorReg<IR::F32>(IR::VectorReg(inst.src[i].code));
        if (IsPosition(attrib)) {
            IR::ExportPosition(ir, runtime_info.hw.vs, tess_emulated_primitive, attrib, i, value);
        } else {
            ir.SetAttribute(attrib, value, i);
        }
    }
}

} // namespace Shader::Gcn
