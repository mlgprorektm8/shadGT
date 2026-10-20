// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <map>
#include <mutex>

#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "shader_recompiler/info.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/shader_usage_serde.h"
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
// 15: shader attribute loads/stores are stored (FIX-012). Version 14 entries still load, with
// their flags recovered on first use by a rect/quad-list pipeline.
// 16: the resource usage FIX-043 keys permutations by follows the Info. Versions 14 and 15
// still load, as translations of unknown usage (keyed on every resource property, as stored).
static constexpr u32 ShaderMetaVersion = 16u;
static constexpr u32 ShaderMetaVersionWithoutResourceUsage = 15u;
static constexpr u32 ShaderMetaVersionWithoutAttributeFlags = 14u;
// 9: rect/quad-list helper shaders built from complete attribute info (FIX-012). Version 8
// entries still load unless they carry such helper shaders, which may have been built from a
// shader without flags.
// 10: quad-list helper control shaders without the local index array (FIX-013). Entries of
// versions 8 and 9 still load unless they carry rect/quad-list helper shaders.
static constexpr u32 PipelineKeyVersion = 10u;
static constexpr u32 PipelineKeyVersionWithoutAttributeFlags = 8u;
static constexpr u32 PipelineKeyVersionOldQuadHelpers = 9u;
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

bool PeekShaderMeta(std::vector<u8>&& blob, u64& pgm_hash, size_t& perm_idx) {
    // The layout LoadShaderMeta reads: versions, permutation hash and index, the
    // specialization, then the shader info, which starts with InfoPersistent.
    Serialization::Archive ar{std::move(blob)};
    Serialization::Reader meta{ar};
    u32 meta_version{};
    meta.Read(meta_version);
    if (meta_version != Serialization::ShaderMetaVersion &&
        meta_version != Serialization::ShaderMetaVersionWithoutResourceUsage &&
        meta_version != Serialization::ShaderMetaVersionWithoutAttributeFlags) {
        return false;
    }
    u32 binary_version{};
    meta.Read(binary_version);
    if (binary_version != Serialization::ShaderBinaryVersion) {
        return false;
    }
    u64 perm_hash{};
    meta.Read(perm_hash);
    meta.Read(perm_idx);
    Shader::StageSpecialization spec{};
    spec.Deserialize(ar);
    alignas(Shader::InfoPersistent) std::array<u8, sizeof(Shader::InfoPersistent)> persistent;
    meta.Read(persistent.data(), persistent.size());
    pgm_hash = reinterpret_cast<const Shader::InfoPersistent*>(persistent.data())->pgm_hash;
    return true;
}

bool LoadStoredGraphicsPipeline(std::vector<u8>&& blob, GraphicsPipelineKey& key,
                                GraphicsPipeline::SerializationSupport& sdata) {
    if (blob.size() < 2 * sizeof(u32) + sizeof(GraphicsPipelineKey)) {
        return false;
    }
    Serialization::Archive ar{std::move(blob)};
    Serialization::Reader pldata{ar};
    u32 version{};
    pldata.Read(version);
    if (version != Serialization::PipelineKeyVersion) {
        return false;
    }
    u32 is_compute{};
    pldata.Read(is_compute);
    if (is_compute != 0) {
        return false;
    }
    key.Deserialize(ar);
    return sdata.Deserialize(ar);
}

namespace {
// SHD-006 census: how many new translations one made before any draw, from a program's code and
// registers alone, would have matched: null sharps (2D/array float textures, unstrided raw
// buffers, plain samplers), pixel inputs in order, plain float targets with the packed export.
// Logged every 250 new translations with what the others needed.
struct TranslationCensus {
    std::mutex mutex;
    u64 total{}, ps{}, ps_default{}, vs{}, vs_default{}, other{};
    std::map<std::string, u64> reasons;
} g_census;

void CountTranslation(const Shader::Info& info, const Shader::StageSpecialization& spec) {
    std::string why;
    const auto add = [&](const char* reason) {
        if (why.find(reason) == std::string::npos) {
            why += why.empty() ? "" : "+";
            why += reason;
        }
    };
    using Type = AmdGpu::ImageType;
    for (const auto& image : spec.images) {
        if (image.type == Type::Cube) {
            add("cube");
        } else if (image.type == Type::Color3D) {
            add("3d");
        } else if (image.type == Type::Color2DMsaa || image.type == Type::Color2DMsaaArray) {
            add("msaa");
        }
        if (image.is_integer) {
            add("int-texture");
        }
        if (image.is_srgb) {
            add("srgb");
        }
        if (image.num_conversion != AmdGpu::NumberConversion::None) {
            add("tex-conversion");
        }
    }
    for (const auto& buffer : spec.buffers) {
        if (buffer.stride != 0) {
            add("stride");
        }
        if (buffer.is_formatted) {
            add("formatted-buffer");
        }
    }
    for (const auto& sampler : spec.samplers) {
        if (sampler.force_unnormalized || sampler.force_degamma) {
            add("sampler");
        }
    }
    const bool is_ps = info.sw_stage == Shader::SwStage::Fragment;
    const bool is_vs = info.sw_stage == Shader::SwStage::Vertex;
    if (is_ps) {
        const auto& fs = spec.runtime_info.hw.fs;
        for (u32 i = 0; i < fs.num_inputs && i < fs.inputs.size(); ++i) {
            if (fs.inputs[i].param_index != i || fs.inputs[i].is_default) {
                add("inputs");
                break;
            }
        }
        for (const auto& cb : fs.color_buffers) {
            if (cb.data_format == AmdGpu::DataFormat::FormatInvalid) {
                continue;
            }
            const bool plain = cb.num_format == AmdGpu::NumberFormat::Float &&
                               cb.num_conversion == AmdGpu::NumberConversion::None &&
                               cb.export_format == AmdGpu::ShaderExportFormat::ABGR_FP16 &&
                               cb.swizzle == AmdGpu::IdentityMapping && !cb.blend_self_scale &&
                               !cb.blend_swizzled_alpha && !cb.blend_swizzled_factors;
            if (!plain) {
                add("outputs");
                break;
            }
        }
    }
    if (is_vs && !spec.vs_attribs.empty()) {
        add("vertex-layout");
    }
    std::scoped_lock lock{g_census.mutex};
    ++g_census.total;
    if (is_ps) {
        ++g_census.ps;
        g_census.ps_default += why.empty() ? 1 : 0;
    } else if (is_vs) {
        ++g_census.vs;
        g_census.vs_default += why.empty() ? 1 : 0;
    } else {
        ++g_census.other;
    }
    if (!why.empty()) {
        ++g_census.reasons[(is_ps ? "ps:" : is_vs ? "vs:" : "other:") + why];
    }
    if (g_census.total % 250 == 0) {
        std::vector<std::pair<u64, std::string>> top;
        for (const auto& [reason, count] : g_census.reasons) {
            top.emplace_back(count, reason);
        }
        std::ranges::sort(top, std::greater{});
        std::string text;
        for (size_t i = 0; i < std::min<size_t>(top.size(), 14); ++i) {
            text += fmt::format(" {}={}", top[i].second, top[i].first);
        }
        LOG_WARNING(Render_Vulkan,
                    "SHD-006 census: {} new translations; pixel {} ({} from code alone), vertex {} "
                    "({} from code alone), other {}; needed:{}",
                    g_census.total, g_census.ps, g_census.ps_default, g_census.vs,
                    g_census.vs_default, g_census.other, text);
    }
}
} // namespace

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash,
                        size_t perm_idx) {
    CountTranslation(info, spec); // SHD-006
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
    WriteResourceUsage(meta, info);

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
    if (meta_version != Serialization::ShaderMetaVersion &&
        meta_version != Serialization::ShaderMetaVersionWithoutResourceUsage &&
        meta_version != Serialization::ShaderMetaVersionWithoutAttributeFlags) {
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
    info.Deserialize(ar, perm_hash_ar,
                     meta_version != Serialization::ShaderMetaVersionWithoutAttributeFlags);
    info.resource_usage_known = false;
    if (meta_version >= Serialization::ShaderMetaVersion) {
        ReadResourceUsage(meta, info);
    }
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

void PipelineCache::Sync() {
    Storage::DataBase::Instance().Close();
}

} // namespace Vulkan

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer info{ar};

    info.Write(this, sizeof(InfoPersistent));
    info.Write(flattened_ud_buf);
    // FIX-012: rect/quad-list helper shaders forward the vertex outputs the fragment shader
    // reads; a preloaded shader without these flags made them forward nothing.
    info.Write(loads.flags.data(), loads.flags.size());
    info.Write(stores.flags.data(), stores.flags.size());
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar, u64 walker_key, bool has_attribute_flags) {
    Serialization::Reader info{ar};

    info.Read(this, sizeof(Shader::InfoPersistent));
    info.Read(flattened_ud_buf);
    attribute_flags_known = has_attribute_flags;
    if (has_attribute_flags) {
        info.Read(loads.flags.data(), loads.flags.size());
        info.Read(stores.flags.data(), stores.flags.size());
    }

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
