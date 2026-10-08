// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <sirit/sirit.h>
#include "shader_recompiler/backend/spirv/emit_spirv_quad_rect.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/runtime_info.h"

namespace Shader::Backend::SPIRV {

using Sirit::Id;

constexpr u32 SPIRV_VERSION_1_5 = 0x00010500;

struct QuadRectListEmitter : public Sirit::Module {
    explicit QuadRectListEmitter(const HwFragmentRuntimeInfo& fs_info, const Info& vertex_info,
                                 const Info* fragment_info)
        : Sirit::Module{SPIRV_VERSION_1_5},
          exports_layer{vertex_info.stores.GetAny(IR::Attribute::RenderTargetIndex)} {
        void_id = TypeVoid();
        bool_id = TypeBool();
        float_id = TypeFloat(32);
        uint_id = TypeUInt(32U);
        int_id = TypeInt(32U, true);
        bvec2_id = TypeVector(bool_id, 2);
        vec2_id = TypeVector(float_id, 2);
        vec3_id = TypeVector(float_id, 3);
        vec4_id = TypeVector(float_id, 4);
        float_types = {Id{}, float_id, vec2_id, vec3_id, vec4_id};

        float_one = Constant(float_id, 1.0f);
        float_min_one = Constant(float_id, -1.0f);
        int_zero = Constant(int_id, 0);

        const Id float_arr{TypeArray(float_id, Constant(uint_id, 1U))};
        gl_per_vertex_type = TypeStruct(vec4_id, float_id, float_arr, float_arr);
        Decorate(gl_per_vertex_type, spv::Decoration::Block);
        MemberDecorate(gl_per_vertex_type, 0U, spv::Decoration::BuiltIn,
                       static_cast<u32>(spv::BuiltIn::Position));
        MemberDecorate(gl_per_vertex_type, 1U, spv::Decoration::BuiltIn,
                       static_cast<u32>(spv::BuiltIn::PointSize));
        MemberDecorate(gl_per_vertex_type, 2U, spv::Decoration::BuiltIn,
                       static_cast<u32>(spv::BuiltIn::ClipDistance));
        MemberDecorate(gl_per_vertex_type, 3U, spv::Decoration::BuiltIn,
                       static_cast<u32>(spv::BuiltIn::CullDistance));

        // This runtime input count is the guest count. The fragment compiler appends its
        // synthetic clip input to a local copy, so it must be added separately here.
        const bool imports_clip = fs_info.clip_distance_emulation &&
                                  vertex_info.stores.GetAny(IR::Attribute::ClipDistance);
        const u32 layer_offset = exports_layer ? 1 : 0;
        if (fragment_info) {
            for (u32 i = 0; i < fs_info.num_inputs; ++i) {
                const auto& input = fs_info.inputs[i];
                const auto fragment_param = IR::Attribute(u32(IR::Attribute::Param0) + i);
                if (input.IsDefault() || !fragment_info->loads.GetAny(fragment_param)) {
                    continue;
                }
                const auto vertex_param =
                    IR::Attribute(u32(IR::Attribute::Param0) + input.param_index);
                const bool imported = vertex_info.stores.GetAny(vertex_param);
                const u32 components = imported
                                           ? vertex_info.stores.NumComponents(vertex_param)
                                           : fragment_info->loads.NumComponents(fragment_param);
                const u32 location = input.param_index + (fs_info.clip_distance_emulation ? 1 : 0);
                // Multiple fragment inputs can select the same vertex parameter.
                if (std::ranges::any_of(params, [location](const Param& param) {
                        return param.location == location;
                    })) {
                    continue;
                }
                params.push_back(
                    {.location = location,
                     .input_location = input.param_index + (imports_clip ? 1 : 0) + layer_offset,
                     .components = components,
                     .imported = imported});
            }
        }
        if (fs_info.clip_distance_emulation) {
            params.push_back({.location = 0,
                              .input_location = layer_offset,
                              .components = MaxEmulatedClipDistances,
                              .imported = imports_clip});
        }
    }

    /// Emits tessellation control shader for interpolating the 4th vertex of rectange primitive
    void EmitRectListTCS() {
        DefineEntry(spv::ExecutionModel::TessellationControl);

        // Set passthrough tessellation factors
        const Id output_float_id{TypePointer(spv::StorageClass::Output, float_id)};
        for (int i = 0; i < 4; i++) {
            const Id ptr{OpAccessChain(output_float_id, gl_tess_level_outer, Int(i))};
            OpStore(ptr, float_one);
        }
        for (int i = 0; i < 2; i++) {
            const Id ptr{OpAccessChain(output_float_id, gl_tess_level_inner, Int(i))};
            OpStore(ptr, float_one);
        }

        const Id input_vec4{TypePointer(spv::StorageClass::Input, vec4_id)};
        const Id output_vec4{TypePointer(spv::StorageClass::Output, vec4_id)};

        // Emit interpolation block of the 4th vertex in rect.
        // Load positions
        std::array<Id, 3> pos;
        for (int i = 0; i < 3; i++) {
            pos[i] = OpLoad(vec4_id, OpAccessChain(input_vec4, gl_in, Int(i), int_zero));
        }

        std::array<Id, 3> point_coord_equal;
        for (int i = 0; i < 3; i++) {
            // point_coord_equal[i] = equal(gl_in[i].gl_Position.xy, gl_in[(i + 1) %
            // 3].gl_Position.xy);
            const Id pos_l_xy{OpVectorShuffle(vec2_id, pos[i], pos[i], 0, 1)};
            const Id pos_r_xy{OpVectorShuffle(vec2_id, pos[(i + 1) % 3], pos[(i + 1) % 3], 0, 1)};
            point_coord_equal[i] = OpFOrdEqual(bvec2_id, pos_l_xy, pos_r_xy);
        }

        std::array<Id, 3> bary_coord;
        std::array<Id, 3> is_edge_vertex;
        for (int i = 0; i < 3; i++) {
            // bool xy_equal = point_coord_equal[i].x && point_coord_equal[(i + 2) % 3].y;
            const Id xy_equal{
                OpLogicalAnd(bool_id, OpCompositeExtract(bool_id, point_coord_equal[i], 0),
                             OpCompositeExtract(bool_id, point_coord_equal[(i + 2) % 3], 1))};
            // bool yx_equal = point_coord_equal[i].y && point_coord_equal[(i + 2) % 3].x;
            const Id yx_equal{
                OpLogicalAnd(bool_id, OpCompositeExtract(bool_id, point_coord_equal[i], 1),
                             OpCompositeExtract(bool_id, point_coord_equal[(i + 2) % 3], 0))};
            // bary_coord[i] = (xy_equal || yx_equal) ? -1.f : 1.f;
            is_edge_vertex[i] = OpLogicalOr(bool_id, xy_equal, yx_equal);
            bary_coord[i] = OpSelect(float_id, is_edge_vertex[i], float_min_one, float_one);
        }

        const auto interpolate = [&](Id type, Id v0, Id v1, Id v2) {
            // return v0 * bary_coord.x + v1 * bary_coord.y + v2 * bary_coord.z;
            const auto multiply = [&](Id value, Id weight) {
                return type.value == float_id.value ? OpFMul(type, value, weight)
                                                    : OpVectorTimesScalar(type, value, weight);
            };
            const Id p0{multiply(v0, bary_coord[0])};
            const Id p1{multiply(v1, bary_coord[1])};
            const Id p2{multiply(v2, bary_coord[2])};
            return OpFAdd(type, p0, OpFAdd(type, p1, p2));
        };

        // int vertex_index_id = is_edge_vertex[1] ? 1 : (is_edge_vertex[2] ? 2 : 0);
        Id vertex_index{OpSelect(int_id, is_edge_vertex[2], Int(2), Int(0))};
        vertex_index = OpSelect(int_id, is_edge_vertex[1], Int(1), vertex_index);

        // int index = (vertex_index_id + gl_InvocationID) % 3;
        const Id invocation_id{OpLoad(int_id, gl_invocation_id)};
        const Id invocation_3{OpIEqual(bool_id, invocation_id, Int(3))};
        const Id index{OpSMod(int_id, OpIAdd(int_id, vertex_index, invocation_id), Int(3))};

        // gl_out[gl_InvocationID].gl_Position = gl_InvocationID == 3 ? pos3 :
        // gl_in[index].gl_Position;
        const Id pos3{interpolate(vec4_id, pos[0], pos[1], pos[2])};
        const Id in_ptr{OpAccessChain(input_vec4, gl_in, index, Int(0))};
        const Id position{OpSelect(vec4_id, invocation_3, pos3, OpLoad(vec4_id, in_ptr))};
        OpStore(OpAccessChain(output_vec4, gl_out, invocation_id, Int(0)), position);

        EmitLayerPassthrough(invocation_id, index);

        // Set attributes
        for (const auto& param : params) {
            const Id type{float_types[param.components]};
            const Id input_pointer{TypePointer(spv::StorageClass::Input, type)};
            const Id output_pointer{TypePointer(spv::StorageClass::Output, type)};
            if (!param.imported) {
                // A guest input without a preceding export is undefined. Keep the linked
                // interface valid without reading a nonexistent vertex output.
                OpStore(OpAccessChain(output_pointer, param.output, invocation_id), OpUndef(type));
                continue;
            }
            // vec4 in_paramN3 = interpolate(bary_coord, in_paramN[0], in_paramN[1], in_paramN[2]);
            const Id v0{OpLoad(type, OpAccessChain(input_pointer, param.input, Int(0)))};
            const Id v1{OpLoad(type, OpAccessChain(input_pointer, param.input, Int(1)))};
            const Id v2{OpLoad(type, OpAccessChain(input_pointer, param.input, Int(2)))};
            const Id in_param3{interpolate(type, v0, v1, v2)};
            // out_paramN[gl_InvocationID] = gl_InvocationID == 3 ? in_paramN3 : in_paramN[index];
            const Id in_param{OpLoad(type, OpAccessChain(input_pointer, param.input, index))};
            const Id out_param{OpSelect(type, invocation_3, in_param3, in_param)};
            OpStore(OpAccessChain(output_pointer, param.output, invocation_id), out_param);
        }

        OpReturn();
        OpFunctionEnd();
    }

    /// Emits a passthrough quad tessellation control shader that outputs 4 control points.
    void EmitQuadListTCS() {
        DefineEntry(spv::ExecutionModel::TessellationControl);

        // Set passthrough tessellation factors
        const Id output_float{TypePointer(spv::StorageClass::Output, float_id)};
        for (int i = 0; i < 4; i++) {
            const Id ptr{OpAccessChain(output_float, gl_tess_level_outer, Int(i))};
            OpStore(ptr, float_one);
        }
        for (int i = 0; i < 2; i++) {
            const Id ptr{OpAccessChain(output_float, gl_tess_level_inner, Int(i))};
            OpStore(ptr, float_one);
        }

        const Id input_vec4{TypePointer(spv::StorageClass::Input, vec4_id)};
        const Id output_vec4{TypePointer(spv::StorageClass::Output, vec4_id)};
        const Id invocation_id{OpLoad(int_id, gl_invocation_id)};
        // Control point i takes input vertex {1, 2, 0, 3}[i]. FIX-013: computed instead of read
        // from an initialized function-local array. In GT Sport captures (NVIDIA) some patches
        // got a first control point matching no vertex shader output, stretching grass and
        // terrain quads across the screen; the array lookup is the suspected cause.
        const Id rotated{OpSMod(int_id, OpIAdd(int_id, invocation_id, Int(1)), Int(3))};
        const Id index{
            OpSelect(int_id, OpIEqual(bool_id, invocation_id, Int(3)), Int(3), rotated)};

        // gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
        const Id in_position{OpLoad(vec4_id, OpAccessChain(input_vec4, gl_in, index, Int(0)))};
        OpStore(OpAccessChain(output_vec4, gl_out, invocation_id, Int(0)), in_position);

        EmitLayerPassthrough(invocation_id, index);

        for (const auto& param : params) {
            const Id type{float_types[param.components]};
            const Id input_pointer{TypePointer(spv::StorageClass::Input, type)};
            const Id output_pointer{TypePointer(spv::StorageClass::Output, type)};
            // out_paramN[gl_InvocationID] = in_paramN[gl_InvocationID];
            const Id in_param{param.imported
                                  ? OpLoad(type, OpAccessChain(input_pointer, param.input, index))
                                  : OpUndef(type)};
            OpStore(OpAccessChain(output_pointer, param.output, invocation_id), in_param);
        }

        OpReturn();
        OpFunctionEnd();
    }

    /// Emits a passthrough quad tessellation evaluation shader that outputs 4 control points.
    void EmitPassthroughTES() {
        DefineEntry(spv::ExecutionModel::TessellationEvaluation);

        // const int index = int(gl_TessCoord.y) * 2 + int(gl_TessCoord.x);
        const Id input_float{TypePointer(spv::StorageClass::Input, float_id)};
        const Id tess_coord_x{OpLoad(float_id, OpAccessChain(input_float, gl_tess_coord, Int(0)))};
        const Id tess_coord_y{OpLoad(float_id, OpAccessChain(input_float, gl_tess_coord, Int(1)))};
        const Id index{OpIAdd(int_id, OpIMul(int_id, OpConvertFToS(int_id, tess_coord_y), Int(2)),
                              OpConvertFToS(int_id, tess_coord_x))};

        // gl_Position = gl_in[index].gl_Position;
        const Id input_vec4{TypePointer(spv::StorageClass::Input, vec4_id)};
        const Id output_vec4{TypePointer(spv::StorageClass::Output, vec4_id)};
        const Id position{OpLoad(vec4_id, OpAccessChain(input_vec4, gl_in, index, Int(0)))};
        OpStore(OpAccessChain(output_vec4, gl_per_vertex, Int(0)), position);

        if (exports_layer) {
            const Id input_uint{TypePointer(spv::StorageClass::Input, uint_id)};
            OpStore(layer_output, OpLoad(uint_id, OpAccessChain(input_uint, layer_input, Int(0))));
        }

        // out_paramN = in_paramN[index];
        for (const auto& param : params) {
            const Id type{float_types[param.components]};
            const Id input_pointer{TypePointer(spv::StorageClass::Input, type)};
            const Id value{OpLoad(type, OpAccessChain(input_pointer, param.input, index))};
            OpStore(param.output, value);
        }

        OpReturn();
        OpFunctionEnd();
    }

private:
    void EmitLayerPassthrough(Id invocation, Id index) {
        if (!exports_layer) {
            return;
        }
        // A layer is constant across a primitive. Preserve it without the floating-point
        // interpolation used for the synthetic rectangle corner.
        const Id input_uint{TypePointer(spv::StorageClass::Input, uint_id)};
        const Id output_uint{TypePointer(spv::StorageClass::Output, uint_id)};
        const Id layer{OpLoad(uint_id, OpAccessChain(input_uint, layer_input, index))};
        OpStore(OpAccessChain(output_uint, layer_output, invocation), layer);
    }

    Id Int(s32 value) {
        return Constant(int_id, value);
    }

    Id AddInput(Id type) {
        const Id input{AddGlobalVariable(TypePointer(spv::StorageClass::Input, type),
                                         spv::StorageClass::Input)};
        interfaces.push_back(input);
        return input;
    }

    Id AddOutput(Id type) {
        const Id output{AddGlobalVariable(TypePointer(spv::StorageClass::Output, type),
                                          spv::StorageClass::Output)};
        interfaces.push_back(output);
        return output;
    }

    void DefineEntry(spv::ExecutionModel model) {
        AddCapability(spv::Capability::Shader);
        AddCapability(spv::Capability::Tessellation);
        const Id void_function{TypeFunction(void_id)};
        main = OpFunction(void_id, spv::FunctionControlMask::MaskNone, void_function);
        if (model == spv::ExecutionModel::TessellationControl) {
            AddExecutionMode(main, spv::ExecutionMode::OutputVertices, 4U);
        } else {
            AddExecutionMode(main, spv::ExecutionMode::Quads);
            AddExecutionMode(main, spv::ExecutionMode::SpacingEqual);
            AddExecutionMode(main, spv::ExecutionMode::VertexOrderCw);
        }
        DefineInputs(model);
        DefineOutputs(model);
        AddEntryPoint(model, main, "main", interfaces);
        AddLabel(OpLabel());
    }

    void DefineOutputs(spv::ExecutionModel model) {
        if (exports_layer) {
            if (model == spv::ExecutionModel::TessellationControl) {
                layer_output = AddOutput(TypeArray(uint_id, Int(4)));
                Decorate(layer_output, spv::Decoration::Location, AuxLayerLocation);
            } else {
                AddCapability(spv::Capability::ShaderLayer);
                layer_output = AddOutput(uint_id);
                Decorate(layer_output, spv::Decoration::BuiltIn, spv::BuiltIn::Layer);
            }
        }
        if (model == spv::ExecutionModel::TessellationControl) {
            const Id gl_per_vertex_array{TypeArray(gl_per_vertex_type, Constant(uint_id, 4U))};
            gl_out = AddOutput(gl_per_vertex_array);

            const Id arr2_id{TypeArray(float_id, Constant(uint_id, 2U))};
            gl_tess_level_inner = AddOutput(arr2_id);
            Decorate(gl_tess_level_inner, spv::Decoration::BuiltIn, spv::BuiltIn::TessLevelInner);
            Decorate(gl_tess_level_inner, spv::Decoration::Patch);

            const Id arr4_id{TypeArray(float_id, Constant(uint_id, 4U))};
            gl_tess_level_outer = AddOutput(arr4_id);
            Decorate(gl_tess_level_outer, spv::Decoration::BuiltIn, spv::BuiltIn::TessLevelOuter);
            Decorate(gl_tess_level_outer, spv::Decoration::Patch);
        } else {
            gl_per_vertex = AddOutput(gl_per_vertex_type);
        }
        for (auto& param : params) {
            const Id type{float_types[param.components]};
            param.output = AddOutput(
                model == spv::ExecutionModel::TessellationControl ? TypeArray(type, Int(4)) : type);
            const u32 layer_offset =
                exports_layer && model == spv::ExecutionModel::TessellationControl ? 1 : 0;
            Decorate(param.output, spv::Decoration::Location, param.location + layer_offset);
        }
    }

    void DefineInputs(spv::ExecutionModel model) {
        if (exports_layer) {
            layer_input = AddInput(TypeArray(uint_id, Int(32)));
            Decorate(layer_input, spv::Decoration::Location, AuxLayerLocation);
        }
        if (model == spv::ExecutionModel::TessellationEvaluation) {
            gl_tess_coord = AddInput(vec3_id);
            Decorate(gl_tess_coord, spv::Decoration::BuiltIn, spv::BuiltIn::TessCoord);
        } else {
            gl_invocation_id = AddInput(int_id);
            Decorate(gl_invocation_id, spv::Decoration::BuiltIn, spv::BuiltIn::InvocationId);
        }
        const Id gl_per_vertex_array{TypeArray(gl_per_vertex_type, Constant(uint_id, 32U))};
        gl_in = AddInput(gl_per_vertex_array);
        for (auto& param : params) {
            const bool evaluation = model == spv::ExecutionModel::TessellationEvaluation;
            if (!evaluation && !param.imported) {
                continue;
            }
            param.input = AddInput(TypeArray(float_types[param.components], Int(32)));
            Decorate(param.input, spv::Decoration::Location,
                     evaluation ? param.location + (exports_layer ? 1 : 0) : param.input_location);
        }
    }

private:
    struct Param {
        u32 location;
        u32 input_location;
        u32 components;
        bool imported;
        Id input;
        Id output;
    };
    std::vector<Param> params;
    bool exports_layer;
    Id layer_input;
    Id layer_output;
    Id main;
    Id void_id;
    Id bool_id;
    Id float_id;
    Id uint_id;
    Id int_id;
    Id bvec2_id;
    Id vec2_id;
    Id vec3_id;
    Id vec4_id;
    std::array<Id, 5> float_types;
    Id float_one;
    Id float_min_one;
    Id int_zero;
    Id gl_per_vertex_type;
    Id gl_in;
    union {
        Id gl_out;
        Id gl_per_vertex;
    };
    Id gl_tess_level_inner;
    Id gl_tess_level_outer;
    union {
        Id gl_tess_coord;
        Id gl_invocation_id;
    };
    std::vector<Id> interfaces;
};

std::vector<u32> EmitAuxilaryTessShader(AuxShaderType type, const HwFragmentRuntimeInfo& fs_info,
                                        const Info& vertex_info, const Info* fragment_info) {
    QuadRectListEmitter ctx{fs_info, vertex_info, fragment_info};
    switch (type) {
    case AuxShaderType::RectListTCS:
        ctx.EmitRectListTCS();
        break;
    case AuxShaderType::QuadListTCS:
        ctx.EmitQuadListTCS();
        break;
    case AuxShaderType::PassthroughTES:
        ctx.EmitPassthroughTES();
        break;
    }
    return ctx.Assemble();
}

} // namespace Shader::Backend::SPIRV
