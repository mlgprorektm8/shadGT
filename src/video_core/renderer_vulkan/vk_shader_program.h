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
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    ModuleList modules{};

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
