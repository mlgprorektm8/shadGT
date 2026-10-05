// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <filesystem>
#include <fstream>
#include <map>
#include <gtest/gtest.h>
#include <spirv/unified1/spirv.hpp11>
#include "shader_recompiler/backend/spirv/emit_spirv_quad_rect.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/runtime_info.h"

using namespace Shader::Backend::SPIRV;

static unsigned CountDecoration(const std::vector<u32>& code, spv::Decoration decoration,
                                u32 value) {
    unsigned count = 0;
    for (size_t i = 5; i < code.size();) {
        const auto words = code[i] >> 16;
        if (words == 0 || i + words > code.size()) {
            ADD_FAILURE() << "Invalid SPIR-V instruction length";
            break;
        }
        if ((code[i] & 0xffff) == u32(spv::Op::OpDecorate) && words == 4 &&
            code[i + 2] == u32(decoration) && code[i + 3] == value) {
            ++count;
        }
        i += words;
    }
    return count;
}

struct ShaderInterface {
    // Location -> component count, after removing the per-vertex array.
    std::map<u32, u32> inputs;
    std::map<u32, u32> outputs;
};

static ShaderInterface ReadInterface(const std::vector<u32>& code) {
    std::map<u32, u32> locations;
    std::map<u32, u32> element_types;
    std::map<u32, u32> components;
    std::map<u32, std::pair<u32, spv::StorageClass>> variables;
    for (size_t i = 5; i < code.size();) {
        const auto words = code[i] >> 16;
        if (words == 0 || i + words > code.size()) {
            ADD_FAILURE() << "Invalid SPIR-V instruction length";
            return {};
        }
        switch (spv::Op(code[i] & 0xffff)) {
        case spv::Op::OpDecorate:
            if (words == 4 && code[i + 2] == u32(spv::Decoration::Location)) {
                locations[code[i + 1]] = code[i + 3];
            }
            break;
        case spv::Op::OpTypePointer:
            element_types[code[i + 1]] = code[i + 3];
            break;
        case spv::Op::OpTypeArray:
            element_types[code[i + 1]] = code[i + 2];
            break;
        case spv::Op::OpTypeVector:
            components[code[i + 1]] = code[i + 3];
            break;
        case spv::Op::OpTypeInt:
        case spv::Op::OpTypeFloat:
            components[code[i + 1]] = 1;
            break;
        case spv::Op::OpVariable:
            variables[code[i + 2]] = {code[i + 1], spv::StorageClass(code[i + 3])};
            break;
        default:
            break;
        }
        i += words;
    }
    ShaderInterface result;
    for (const auto& [variable, location] : locations) {
        const auto [pointer_type, storage] = variables.at(variable);
        u32 type = pointer_type;
        while (element_types.contains(type)) {
            type = element_types.at(type);
        }
        auto& interface = storage == spv::StorageClass::Input ? result.inputs : result.outputs;
        EXPECT_TRUE(interface.emplace(location, components.at(type)).second)
            << "Duplicate interface variable at location " << location;
    }
    return result;
}

static unsigned CountOpcode(const std::vector<u32>& code, spv::Op opcode) {
    unsigned count = 0;
    for (size_t i = 5; i < code.size();) {
        const auto words = code[i] >> 16;
        if (words == 0 || i + words > code.size()) {
            ADD_FAILURE() << "Invalid SPIR-V instruction length";
            break;
        }
        count += (code[i] & 0xffff) == u32(opcode);
        i += words;
    }
    return count;
}

TEST(AuxTessellation, CarriesLayerThroughControlAndWritesItInEvaluation) {
    Shader::HwFragmentRuntimeInfo fs{};
    fs.num_inputs = 1;
    fs.inputs[0].param_index = 2;
    for (const auto type :
         {AuxShaderType::RectListTCS, AuxShaderType::QuadListTCS, AuxShaderType::PassthroughTES}) {
        for (const bool layer : {false, true}) {
            Shader::Info vertex{};
            Shader::Info fragment{};
            vertex.stores.Set(Shader::IR::Attribute::Param2);
            fragment.loads.Set(Shader::IR::Attribute::Param0);
            if (layer) {
                vertex.stores.Set(Shader::IR::Attribute::RenderTargetIndex);
            }
            const auto code = EmitAuxilaryTessShader(type, fs, vertex, &fragment);
            const bool evaluation = type == AuxShaderType::PassthroughTES;
            EXPECT_EQ(CountDecoration(code, spv::Decoration::BuiltIn, u32(spv::BuiltIn::Layer)),
                      layer && evaluation ? 1 : 0);
            EXPECT_EQ(CountDecoration(code, spv::Decoration::Location, AuxLayerLocation),
                      layer ? (evaluation ? 1 : 2) : 0);
            // Keep real modules available for independent spirv-val checks.
            const auto path = std::filesystem::path("aux-tess-" + std::to_string(u32(type)) +
                                                    (layer ? "-layer.spv" : "-plain.spv"));
            std::ofstream output(path, std::ios::binary);
            ASSERT_TRUE(output.is_open());
            output.write(reinterpret_cast<const char*>(code.data()), code.size() * sizeof(u32));
            ASSERT_TRUE(output.good());
        }
    }
}

TEST(AuxTessellation, KeepsLayerSeparateFromInjectedClipDistances) {
    Shader::HwFragmentRuntimeInfo fs{};
    // GraphicsPipeline passes the raw guest count, before the compiler adds its clip input.
    fs.num_inputs = 1;
    fs.inputs[0].param_index = 2;
    fs.clip_distance_emulation = true;
    Shader::Info vertex{};
    Shader::Info fragment{};
    vertex.stores.Set(Shader::IR::Attribute::Param2);
    vertex.stores.Set(Shader::IR::Attribute::ClipDistance);
    vertex.stores.Set(Shader::IR::Attribute::RenderTargetIndex);
    fragment.loads.Set(Shader::IR::Attribute::Param0);
    for (const auto type :
         {AuxShaderType::RectListTCS, AuxShaderType::QuadListTCS, AuxShaderType::PassthroughTES}) {
        const auto code = EmitAuxilaryTessShader(type, fs, vertex, &fragment);
        const bool evaluation = type == AuxShaderType::PassthroughTES;
        EXPECT_EQ(CountDecoration(code, spv::Decoration::Location, 1), evaluation ? 1 : 2);
        EXPECT_EQ(CountDecoration(code, spv::Decoration::Location, 4), evaluation ? 1 : 2);
        EXPECT_EQ(CountDecoration(code, spv::Decoration::Location, 0), 2);
        const auto path =
            std::filesystem::path("aux-tess-" + std::to_string(u32(type)) + "-clip-layer.spv");
        std::ofstream output(path, std::ios::binary);
        ASSERT_TRUE(output.is_open());
        output.write(reinterpret_cast<const char*>(code.data()), code.size() * sizeof(u32));
        ASSERT_TRUE(output.good());
    }
}

TEST(AuxTessellation, ImportsActiveExportsAndKeepsPartialExportsAtTheirDeclaredWidth) {
    Shader::HwFragmentRuntimeInfo fs{};
    fs.num_inputs = 5;
    fs.inputs[0].param_index = 2;
    fs.inputs[1] = {.param_index = 7, .is_default = true};
    fs.inputs[2].param_index = 2;  // Alias the first parameter.
    fs.inputs[3].param_index = 12; // Unused interpolation registers must not become inputs.
    fs.inputs[4].param_index = 15;
    Shader::Info vertex{};
    Shader::Info fragment{};
    vertex.stores.Set(Shader::IR::Attribute::Param2, 0);
    vertex.stores.Set(Shader::IR::Attribute::Param2, 2);
    fragment.loads.Set(Shader::IR::Attribute::Param0, 0);
    fragment.loads.Set(Shader::IR::Attribute::Param1, 0);
    fragment.loads.Set(Shader::IR::Attribute::Param2, 2);
    // The vertex emitter declares vec4 even when only x/z are written.
    const std::map<u32, u32> expected{
        {2, vertex.stores.NumComponents(Shader::IR::Attribute::Param2)}};
    for (const auto type :
         {AuxShaderType::RectListTCS, AuxShaderType::QuadListTCS, AuxShaderType::PassthroughTES}) {
        const auto code = EmitAuxilaryTessShader(type, fs, vertex, &fragment);
        const auto interface = ReadInterface(code);
        EXPECT_EQ(interface.inputs, expected);
        EXPECT_EQ(interface.outputs, expected);
        EXPECT_EQ(CountOpcode(code, spv::Op::OpUndef), 0);
    }
}

TEST(AuxTessellation, MissingVertexExportsDoNotCreateUnmatchedControlInputs) {
    Shader::HwFragmentRuntimeInfo fs{};
    fs.num_inputs = 2;
    fs.inputs[0].param_index = 0;
    fs.inputs[1].param_index = 6;
    Shader::Info vertex{};
    Shader::Info fragment{};
    vertex.stores.Set(Shader::IR::Attribute::Param0, 1);
    fragment.loads.Set(Shader::IR::Attribute::Param0, 1);
    fragment.loads.Set(Shader::IR::Attribute::Param1, 0);
    const std::map<u32, u32> vertex_outputs{{0, 4}};
    const std::map<u32, u32> fragment_inputs{{0, 4}, {6, 4}};
    const auto evaluation =
        ReadInterface(EmitAuxilaryTessShader(AuxShaderType::PassthroughTES, fs, vertex, &fragment));
    for (const auto type : {AuxShaderType::RectListTCS, AuxShaderType::QuadListTCS}) {
        const auto code = EmitAuxilaryTessShader(type, fs, vertex, &fragment);
        const auto control = ReadInterface(code);
        EXPECT_EQ(control.inputs, vertex_outputs);
        EXPECT_EQ(control.outputs, evaluation.inputs);
        EXPECT_EQ(CountOpcode(code, spv::Op::OpUndef), 1);
    }
    EXPECT_EQ(evaluation.outputs, fragment_inputs);
}

TEST(AuxTessellation, ClipShiftMatchesActualVertexExports) {
    Shader::HwFragmentRuntimeInfo fs{};
    fs.num_inputs = 1;
    fs.inputs[0].param_index = 2;
    fs.clip_distance_emulation = true;
    Shader::Info vertex{};
    Shader::Info fragment{};
    vertex.stores.Set(Shader::IR::Attribute::Param2, 0);
    vertex.stores.Set(Shader::IR::Attribute::RenderTargetIndex);
    fragment.loads.Set(Shader::IR::Attribute::Param0, 0);
    // No clip export means the vertex parameter is shifted only by the layer varying.
    const std::map<u32, u32> vertex_outputs{{0, 1}, {3, 4}};
    const std::map<u32, u32> intermediate{{0, 1}, {1, 4}, {4, 4}};
    const std::map<u32, u32> fragment_inputs{{0, 4}, {3, 4}};
    for (const auto type : {AuxShaderType::RectListTCS, AuxShaderType::QuadListTCS}) {
        const auto code = EmitAuxilaryTessShader(type, fs, vertex, &fragment);
        const auto control = ReadInterface(code);
        EXPECT_EQ(control.inputs, vertex_outputs);
        EXPECT_EQ(control.outputs, intermediate);
        EXPECT_EQ(CountOpcode(code, spv::Op::OpUndef), 1);
    }
    const auto evaluation =
        ReadInterface(EmitAuxilaryTessShader(AuxShaderType::PassthroughTES, fs, vertex, &fragment));
    EXPECT_EQ(evaluation.inputs, intermediate);
    EXPECT_EQ(evaluation.outputs, fragment_inputs);
}

TEST(AuxTessellation, CarriesClipDistancesWithoutGuestFragmentShader) {
    Shader::HwFragmentRuntimeInfo fs{};
    fs.clip_distance_emulation = true;
    Shader::Info vertex{};
    vertex.stores.Set(Shader::IR::Attribute::Param0);
    vertex.stores.Set(Shader::IR::Attribute::ClipDistance);
    const std::map<u32, u32> clip_only{{0, Shader::MaxEmulatedClipDistances}};
    for (const auto type :
         {AuxShaderType::RectListTCS, AuxShaderType::QuadListTCS, AuxShaderType::PassthroughTES}) {
        const auto interface = ReadInterface(EmitAuxilaryTessShader(type, fs, vertex, nullptr));
        EXPECT_EQ(interface.inputs, clip_only);
        EXPECT_EQ(interface.outputs, clip_only);
    }
}
