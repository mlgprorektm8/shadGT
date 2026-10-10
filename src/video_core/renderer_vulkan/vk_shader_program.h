// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <memory>
#include <boost/container/small_vector.hpp>
#include "shader_recompiler/specialization.h"
#include "vulkan/vulkan.hpp"

namespace Vulkan {

struct Program {
    struct Module {
        vk::ShaderModule module{};
        Shader::StageSpecialization spec;
        // Pipelines retain this address, so metadata must survive list reallocations.
        std::unique_ptr<Shader::Info> info;
        /// FIX-043 diagnostic: hash of the SPIR-V, to count permutations that repeat another.
        u64 spv_hash{};
        /// PERF-032: listed from the shader store; its module is loaded the first time it matches.
        bool stored{};
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    ModuleList modules{};

    /// PERF-044: the inputs of the last permutation search and its outcome. A specialization is
    /// made from the permutation's info, the runtime info, the start bindings and the sharps,
    /// which come from the user data and the flattened user data (and, for vertex and
    /// tessellation stages, from guest memory too, so those are never memoized). With the same
    /// inputs, permutations up to the one that matched decide the same way again.
    struct MatchMemo {
        bool valid{};
        size_t perm{};
        size_t modules_size{};
        Shader::RuntimeInfo runtime_info{};
        Shader::Backend::Bindings start{};
        std::vector<u32> user_data;
        std::vector<std::vector<u32>> flats;
    } match_memo;

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                   std::unique_ptr<Shader::Info> info) {
        spec.info = info.get();
        modules.emplace_back(module, std::move(spec), std::move(info));
    }

    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      std::unique_ptr<Shader::Info> info, size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1));
        ASSERT(!modules[perm_idx].info);
        spec.info = info.get();
        modules[perm_idx] = {module, std::move(spec), std::move(info)};
    }
};

} // namespace Vulkan
