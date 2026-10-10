# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

# Find or download Google Test
include(FetchContent)
    FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
FetchContent_MakeAvailable(googletest)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 23)
    foreach(t gtest gtest_main gmock gmock_main)
        if (TARGET ${t})
            target_compile_options(${t} PRIVATE -Wno-error=character-conversion)
        endif()
    endforeach()
endif()

add_executable(shadps4_stencil_test test_stencil_reference.cpp)
target_include_directories(shadps4_stencil_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_stencil_test PRIVATE GTest::gtest_main fmt::fmt)
include(GoogleTest)
gtest_discover_tests(shadps4_stencil_test)

add_executable(shadps4_depth_attachment_test test_depth_attachment.cpp)
target_include_directories(shadps4_depth_attachment_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_depth_attachment_test PRIVATE GTest::gtest_main fmt::fmt Vulkan::Headers)
gtest_discover_tests(shadps4_depth_attachment_test)

add_executable(shadps4_color_clear_test test_color_clear.cpp test_color_blend.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/renderer_vulkan/liverpool_to_vk.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/pixel_format.cpp
    ${CMAKE_SOURCE_DIR}/src/common/number_utils.cpp)
target_include_directories(shadps4_color_clear_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_color_clear_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_color_clear_test PRIVATE
    GTest::gtest_main fmt::fmt Boost::headers Vulkan::Headers magic_enum::magic_enum half::half)
gtest_discover_tests(shadps4_color_clear_test)

add_executable(shadps4_buffer_copy_test test_buffer_copy.cpp)
target_include_directories(shadps4_buffer_copy_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_buffer_copy_test PRIVATE
    GTest::gtest_main fmt::fmt Boost::headers Vulkan::Headers)
gtest_discover_tests(shadps4_buffer_copy_test)

add_executable(shadps4_device_service_test test_device_service.cpp
    ${CMAKE_SOURCE_DIR}/src/core/libraries/device_service/device_service.cpp)
target_include_directories(shadps4_device_service_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_device_service_test PRIVATE GTest::gtest_main fmt::fmt)
gtest_discover_tests(shadps4_device_service_test)

add_executable(shadps4_pm4_type0_test test_pm4_type0.cpp)
target_include_directories(shadps4_pm4_type0_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_pm4_type0_test PRIVATE GTest::gtest_main)
gtest_discover_tests(shadps4_pm4_type0_test)

add_executable(shadps4_openpsid_test test_openpsid.cpp
    ${CMAKE_SOURCE_DIR}/src/core/libraries/kernel/openpsid.cpp)
target_include_directories(shadps4_openpsid_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_openpsid_test PRIVATE GTest::gtest_main fmt::fmt)
gtest_discover_tests(shadps4_openpsid_test)

add_executable(shadps4_buffer_resource_test test_buffer_resource.cpp)
target_include_directories(shadps4_buffer_resource_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_buffer_resource_test PRIVATE GTest::gtest_main fmt::fmt)
gtest_discover_tests(shadps4_buffer_resource_test)

add_executable(shadps4_image_resource_test test_image_resource.cpp)
target_include_directories(shadps4_image_resource_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_image_resource_test PRIVATE GTest::gtest_main fmt::fmt Boost::headers)
gtest_discover_tests(shadps4_image_resource_test)

add_executable(shadps4_shader_program_test test_shader_program.cpp)
target_include_directories(shadps4_shader_program_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_shader_program_test PRIVATE
    GTest::gtest_main fmt::fmt Boost::headers Vulkan::Headers tsl::robin_map xxHash::xxhash
    magic_enum::magic_enum)
gtest_discover_tests(shadps4_shader_program_test)

add_executable(shadps4_image_aliasing_test test_image_aliasing.cpp)
target_include_directories(shadps4_image_aliasing_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_image_aliasing_test PRIVATE
    GTest::gtest_main fmt::fmt Boost::headers Vulkan::Headers)
gtest_discover_tests(shadps4_image_aliasing_test)

add_executable(shadps4_image_mips_test test_image_mips.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/texture_cache/image_info.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/texture_cache/host_compatibility.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/tiling.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/pixel_format.cpp)
target_include_directories(shadps4_image_mips_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_image_mips_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_image_mips_test PRIVATE
    GTest::gtest_main fmt::fmt Boost::headers Vulkan::Headers magic_enum::magic_enum)
gtest_discover_tests(shadps4_image_mips_test)

add_executable(shadps4_draw_pipe_test test_draw_pipe.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/draw_pipe.cpp)
target_include_directories(shadps4_draw_pipe_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_draw_pipe_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_draw_pipe_test PRIVATE GTest::gtest_main)
gtest_discover_tests(shadps4_draw_pipe_test)

add_executable(shadps4_uploaded_pages_test test_uploaded_pages.cpp)
target_include_directories(shadps4_uploaded_pages_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_uploaded_pages_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_uploaded_pages_test PRIVATE GTest::gtest_main tsl::robin_map)
gtest_discover_tests(shadps4_uploaded_pages_test)

add_executable(shadps4_read_ahead_states_test test_read_ahead_states.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/read_ahead_states.cpp)
target_include_directories(shadps4_read_ahead_states_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_read_ahead_states_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_read_ahead_states_test PRIVATE GTest::gtest_main xxHash::xxhash)
gtest_discover_tests(shadps4_read_ahead_states_test)

add_executable(shadps4_build_queue_test test_build_queue.cpp)
target_include_directories(shadps4_build_queue_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_build_queue_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_build_queue_test PRIVATE GTest::gtest_main)
gtest_discover_tests(shadps4_build_queue_test)

add_executable(shadps4_guest_clock_test test_guest_clock.cpp)
target_include_directories(shadps4_guest_clock_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_guest_clock_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_guest_clock_test PRIVATE GTest::gtest_main)
gtest_discover_tests(shadps4_guest_clock_test)

add_executable(shadps4_image_barriers_test test_image_barriers.cpp)
target_include_directories(shadps4_image_barriers_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_compile_features(shadps4_image_barriers_test PRIVATE cxx_std_23)
target_link_libraries(shadps4_image_barriers_test PRIVATE
    GTest::gtest_main fmt::fmt Boost::headers Vulkan::Headers magic_enum::magic_enum)
gtest_discover_tests(shadps4_image_barriers_test)

add_executable(shadps4_image_descriptor_test test_image_descriptor.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/pixel_format.cpp)
target_include_directories(shadps4_image_descriptor_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_image_descriptor_test PRIVATE
    GTest::gtest_main fmt::fmt Boost::headers Vulkan::Headers magic_enum::magic_enum)
gtest_discover_tests(shadps4_image_descriptor_test)

add_executable(shadps4_aux_tessellation_test test_aux_tessellation.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_quad_rect.cpp)
target_include_directories(shadps4_aux_tessellation_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_aux_tessellation_test PRIVATE
    GTest::gtest_main fmt::fmt sirit Boost::headers magic_enum::magic_enum)
gtest_discover_tests(shadps4_aux_tessellation_test)

add_executable(shadps4_resource_binding_test test_resource_binding.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/pixel_format.cpp)
target_include_directories(shadps4_resource_binding_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_resource_binding_test PRIVATE
    GTest::gtest_main fmt::fmt Boost::headers Vulkan::Headers magic_enum::magic_enum)
gtest_discover_tests(shadps4_resource_binding_test)

add_executable(shadps4_shared_memory_storage_test test_shared_memory_storage.cpp)
target_include_directories(shadps4_shared_memory_storage_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_shared_memory_storage_test PRIVATE
    GTest::gtest_main fmt::fmt Boost::headers magic_enum::magic_enum)
gtest_discover_tests(shadps4_shared_memory_storage_test)

add_executable(shadps4_ce_de_counter_test test_ce_de_counter.cpp)
target_include_directories(shadps4_ce_de_counter_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_ce_de_counter_test PRIVATE GTest::gtest_main)
gtest_discover_tests(shadps4_ce_de_counter_test)

add_executable(shadps4_pm4_rewind_test test_pm4_rewind.cpp)
target_include_directories(shadps4_pm4_rewind_test PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(shadps4_pm4_rewind_test PRIVATE GTest::gtest_main)
gtest_discover_tests(shadps4_pm4_rewind_test)

set(TEST_TARGETS
    shadps4_settings_test
    shadps4_ngs2_test
    shadps4_gcn_test
)

set(SETTINGS_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp

    # Minimal common support
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/classes.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/core_stub.cpp
    stubs/signals_stub.cpp
    stubs/file_sys_stub.cpp
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp

    # Tests
    test_emulator_settings.cpp
)

set(NGS2_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/libraries/ngs2/ngs2_impl.cpp

    # Minimal common and kernel support
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp
    ${CMAKE_SOURCE_DIR}/src/core/libraries/kernel/sync/mutex.cpp
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/classes.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/core_stub.cpp
    stubs/signals_stub.cpp
    stubs/file_sys_stub.cpp
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp

    # Tests
    test_ngs2.cpp
)
set(GCN_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_atomic.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_barriers.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_bitwise_conversion.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_composite.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_context_get_set.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_convert.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_floating_point.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_image.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_integer.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_logical.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_select.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_shared_memory.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_special.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_undefined.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_warp.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/emit_spirv_instructions.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/spirv_emit_context.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/backend/spirv/spirv_emit_context.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/format.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/decode.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/decode.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/fetch_shader.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/fetch_shader.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/data_share.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/export.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/translate.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/translate.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/scalar_alu.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/scalar_flow.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/scalar_memory.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/vector_alu.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/vector_interpolation.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/frontend/translate/vector_memory.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/abstract_syntax_list.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/abstract_syntax_list.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/basic_block.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/basic_block.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/microinstruction.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/attribute.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/attribute.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/ir_emitter.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/ir_emitter.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/opcodes.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/opcodes.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/patch.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/patch.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/program.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/program.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/ir_passes.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/constant_propagation_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/dead_code_elimination_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/shader_info_collection_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/resource_discover_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/flatten_extended_userdata_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/resource_patching_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/passes/ssa_rewrite_pass.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/post_order.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/post_order.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/type.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/type.h
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/value.cpp
    ${CMAKE_SOURCE_DIR}/src/shader_recompiler/ir/value.h
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/pixel_format.cpp
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/pixel_format.h
    ${CMAKE_SOURCE_DIR}/src/video_core/amdgpu/resource.h

    # Minimal common support
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/io_file.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/classes.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp
    ${CMAKE_SOURCE_DIR}/src/common/ntapi.cpp
    ${CMAKE_SOURCE_DIR}/src/common/decoder.cpp
    ${CMAKE_SOURCE_DIR}/src/common/signal_context.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/resource_tracking_pass_stub.cpp
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp
    stubs/core_stub.cpp
    stubs/signals_stub.cpp
    stubs/file_sys_stub.cpp

    gcn/gcn_test_runner.hpp
    gcn/gcn_test_runner.cpp
    gcn/translator.hpp
    gcn/translator.cpp

    # Tests
    gcn/test_gcn_instructions.cpp
    test_dynamic_readconst.cpp
    test_resource_patching.cpp
    test_color_export.cpp
    test_srt_walker_registry.cpp
)

add_executable(shadps4_settings_test ${SETTINGS_TEST_SOURCES})
add_executable(shadps4_ngs2_test ${NGS2_TEST_SOURCES})
add_executable(shadps4_gcn_test ${GCN_TEST_SOURCES})

foreach(t ${TEST_TARGETS})
    target_include_directories(${t} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}
    )
    target_compile_features(${t} PRIVATE cxx_std_23)
    target_compile_definitions(${t} PRIVATE BOOST_ASIO_STANDALONE)
endforeach()

target_link_libraries(shadps4_settings_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    nlohmann_json::nlohmann_json
    toml11::toml11
    SDL3::SDL3
    spdlog::spdlog
)
target_link_libraries(shadps4_ngs2_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    nlohmann_json::nlohmann_json
    toml11::toml11
    SDL3::SDL3
    spdlog::spdlog
)
target_link_libraries(shadps4_gcn_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    tsl::robin_map
    nlohmann_json::nlohmann_json
    magic_enum::magic_enum
    toml11::toml11
    Boost::headers
    Vulkan::Headers
    sirit
    SDL3::SDL3
    spdlog::spdlog
    half::half
    xbyak::xbyak
    Zydis::Zydis
)


foreach(t ${TEST_TARGETS})
    if (WIN32)
        target_link_libraries(${t} PRIVATE onecore)
        target_compile_definitions(${t} PRIVATE
            NOMINMAX
            WIN32_LEAN_AND_MEAN
            NTDDI_VERSION=0x0A000006
            _WIN32_WINNT=0x0A00
            WINVER=0x0A00
        )
        if (MSVC)
            target_compile_definitions(${t} PRIVATE
                _CRT_SECURE_NO_WARNINGS
                _CRT_NONSTDC_NO_DEPRECATE
                _SCL_SECURE_NO_WARNINGS
                _TIMESPEC_DEFINED
            )
        endif()
    endif()
endforeach()

include(GoogleTest)

foreach(t ${TEST_TARGETS})
    gtest_discover_tests(${t}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DISCOVERY_TIMEOUT 60
    )
endforeach()

# ===========================================================================
# HTTP tests (libSceHttp - URI helpers and status-line parsing)
# ===========================================================================
# Self-contained block - mirrors the structure of the targets above so the
# new test does not depend on any earlier foreach iterations.

set(HTTP_TEST_SOURCES
    # Under test
    ${CMAKE_SOURCE_DIR}/src/core/libraries/network/http.cpp
    # Required to link RegisterLib's LIB_FUNCTION calls and the logger's
    # access to EmulatorSettings.
    ${CMAKE_SOURCE_DIR}/src/core/emulator_settings.cpp
    ${CMAKE_SOURCE_DIR}/src/core/emulator_state.cpp

    # Minimal common support
    ${CMAKE_SOURCE_DIR}/src/common/path_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/assert.cpp
    ${CMAKE_SOURCE_DIR}/src/common/error.cpp
    ${CMAKE_SOURCE_DIR}/src/common/string_util.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/classes.cpp
    ${CMAKE_SOURCE_DIR}/src/common/logging/log.cpp

    # Stubs that replace dependencies
    stubs/common_stub.cpp
    stubs/scm_rev_stub.cpp
    stubs/sdl_stub.cpp
    stubs/loader_stub.cpp
    stubs/core_stub.cpp
    stubs/signals_stub.cpp
    stubs/file_sys_stub.cpp
    stubs/kernel_stub.cpp

    # Tests
    network/test_http_uri.cpp
    network/test_http_status_line.cpp
    network/test_http_parse_response_header.cpp
    network/test_http_lifecycle.cpp
)

add_executable(shadps4_http_test ${HTTP_TEST_SOURCES})

list(APPEND TEST_TARGETS shadps4_http_test)

target_include_directories(shadps4_http_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}
)
target_compile_features(shadps4_http_test PRIVATE cxx_std_23)
target_compile_definitions(shadps4_http_test PRIVATE BOOST_ASIO_STANDALONE)

target_link_libraries(shadps4_http_test PRIVATE
    GTest::gtest_main
    fmt::fmt
    magic_enum::magic_enum
    nlohmann_json::nlohmann_json
    toml11::toml11
    SDL3::SDL3
    spdlog::spdlog
    ZLIB::ZLIB
)


if (WIN32)
    target_link_libraries(shadps4_http_test PRIVATE onecore)
    target_compile_definitions(shadps4_http_test PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        NTDDI_VERSION=0x0A000006
        _WIN32_WINNT=0x0A00
        WINVER=0x0A00
    )
    if (MSVC)
        target_compile_definitions(shadps4_http_test PRIVATE
            _CRT_SECURE_NO_WARNINGS
            _CRT_NONSTDC_NO_DEPRECATE
            _SCL_SECURE_NO_WARNINGS
            _TIMESPEC_DEFINED
        )
    endif()
endif()

gtest_discover_tests(shadps4_http_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DISCOVERY_TIMEOUT 60
)

# ===========================================================================
# shadGT: offline wave64 lowering inspection of dumped GCN compute shaders
# ===========================================================================
file(GLOB_RECURSE WAVE64_INSPECT_RECOMPILER ${CMAKE_SOURCE_DIR}/src/shader_recompiler/*.cpp)
set(WAVE64_INSPECT_SOURCES ${GCN_TEST_SOURCES})
list(FILTER WAVE64_INSPECT_SOURCES EXCLUDE REGEX "/shader_recompiler/|gcn/|test_|resource_tracking_pass_stub")
add_executable(shadgt_wave64_inspect tools/wave64_inspect.cpp ${WAVE64_INSPECT_RECOMPILER}
               ${WAVE64_INSPECT_SOURCES})
target_include_directories(shadgt_wave64_inspect PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
target_compile_features(shadgt_wave64_inspect PRIVATE cxx_std_23)
target_link_libraries(shadgt_wave64_inspect PRIVATE fmt::fmt nlohmann_json::nlohmann_json
    magic_enum::magic_enum toml11::toml11 Boost::headers Vulkan::Headers sirit SDL3::SDL3
    tsl::robin_map
    spdlog::spdlog half::half xbyak::xbyak Zydis::Zydis)
if (WIN32)
    target_link_libraries(shadgt_wave64_inspect PRIVATE onecore)
    target_compile_definitions(shadgt_wave64_inspect PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN
        NTDDI_VERSION=0x0A000006 _WIN32_WINNT=0x0A00 WINVER=0x0A00 _CRT_SECURE_NO_WARNINGS)
endif()
