// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/serdes.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

void RegisterPipelineData(const ComputePipelineKey& key,
                          ComputePipeline::SerializationSupport& sdata);
void RegisterPipelineData(const GraphicsPipelineKey& key, u64 hash,
                          GraphicsPipeline::SerializationSupport& sdata);
void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash, size_t perm_idx);
void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx);
bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    Shader::StageSpecialization& spec, size_t& perm_idx);
/// PERF-038: the program hash and permutation index of stored shader metadata, without decoding
/// the rest (decoding registers SRT walker code, which only the GPU thread may do).
bool PeekShaderMeta(std::vector<u8>&& blob, u64& pgm_hash, size_t& perm_idx);
/// PERF-038: a stored graphics pipeline of the current key version (others are skipped).
bool LoadStoredGraphicsPipeline(std::vector<u8>&& blob, GraphicsPipelineKey& key,
                                GraphicsPipeline::SerializationSupport& sdata);

} // namespace Vulkan
