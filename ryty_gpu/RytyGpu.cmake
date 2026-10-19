# Builds shadGT's HLE libraries as Ryty host libraries (libs/*.prx).
#
# - libSceVideoOut.prx: shadGT's video output, Gnm driver and Vulkan renderer. It also exports the
#   libSceGnmDriver NIDs; Ryty's loader searches every loaded library for each NID.
# - libSceNgs2.prx: shadGT's Ngs2 audio library, replacing Ryty's PS5 one for PS4 titles.

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(RYTY_GPU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ryty_gpu")

# Emulator-core pieces every library needs, with Ryty stand-ins for the rest.
set(RYTY_COMMON_SOURCES
    ${COMMON}
    src/core/emulator_settings.cpp
    src/core/emulator_state.cpp
    src/core/user_settings.cpp
    src/core/user_manager.cpp
    src/core/file_format/psf.cpp
    src/core/signals.cpp
    src/core/libraries/kernel/sync/mutex.cpp
    ryty_gpu/ryty_library.cpp
    ryty_gpu/ryty_core_shims.cpp
    ryty_gpu/ryty_stubs.cpp
    src/core/file_format/npbind.cpp
)
# shadGT's Discord presence is not part of the Ryty build.
list(FILTER RYTY_COMMON_SOURCES EXCLUDE REGEX "discord_rpc_handler")

set(RYTY_COMMON_LIBRARIES spdlog::spdlog magic_enum::magic_enum fmt::fmt toml11::toml11 tsl::robin_map
    xbyak::xbyak Tracy::TracyClient RenderDoc::API Dear_ImGui gcn half::half Boost::headers
    GPUOpen::VulkanMemoryAllocator sirit Vulkan::Headers xxHash::xxhash Zydis::Zydis SDL3::SDL3
    pugixml::pugixml stb::headers nlohmann_json::nlohmann_json ZLIB::ZLIB PNG::PNG Freetype::Freetype
    miniz::miniz res::embedded absl::flat_hash_map absl::flat_hash_set)

# ryty_add_library(<target> <output name> EXPORT_SOURCES <files registering NIDs> SOURCES <files>
#                  [HLE_HEADER <header> HLE_REGISTER <RegisterLib> LOG <log file>])
function(ryty_add_library target output)
    cmake_parse_arguments(RYTY "" "HLE_HEADER;HLE_REGISTER;LOG" "SOURCES;EXPORT_SOURCES" ${ARGN})
    set(generated "${CMAKE_CURRENT_BINARY_DIR}/ryty_gpu/${target}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${RYTY_GPU_DIR}/gen_exports.py" "${CMAKE_CURRENT_SOURCE_DIR}/src"
                "${generated}" "${output}.prx" ${RYTY_EXPORT_SOURCES}
        COMMAND_ERROR_IS_FATAL ANY)
    foreach(source ${RYTY_EXPORT_SOURCES})
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/${source}")
    endforeach()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${RYTY_GPU_DIR}/gen_exports.py")

    add_library(${target} SHARED ${RYTY_SOURCES} ${RYTY_COMMON_SOURCES}
        "${generated}/ryty_exports.cpp" "${generated}/ryty_exports.def")
    if (RYTY_HLE_REGISTER)
        target_sources(${target} PRIVATE ryty_gpu/ryty_hle_entry.cpp ryty_gpu/ryty_hle_stubs.cpp)
        target_compile_definitions(${target} PRIVATE RYTY_HLE_HEADER="${RYTY_HLE_HEADER}"
            RYTY_HLE_REGISTER=${RYTY_HLE_REGISTER} RYTY_HLE_LOG="${RYTY_LOG}")
    endif()
    set_target_properties(${target} PROPERTIES OUTPUT_NAME ${output} SUFFIX .prx PREFIX "")
    target_link_libraries(${target} PRIVATE ${RYTY_COMMON_LIBRARIES})
    target_compile_definitions(${target} PRIVATE IMGUI_USER_CONFIG="imgui/imgui_config.h" RYTY_GPU=1)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}
        ${HOST_SHADERS_INCLUDE} ${IMGUI_RESOURCES_INCLUDE})
    add_dependencies(${target} host_shaders ImGui_Resources)
    if (WIN32)
        target_link_libraries(${target} PRIVATE mincore wbemuuid)
        target_link_options(${target} PRIVATE /errorlimit:0)
        # Line tables and a PDB, so crash and assertion stacks (module+offset) can be symbolized.
        target_compile_options(${target} PRIVATE /clang:-gline-tables-only /Z7)
        target_link_options(${target} PRIVATE /DEBUG:FULL)
        if (MSVC)
            target_link_libraries(${target} PRIVATE clang_rt.builtins-x86_64.lib)
        endif()
    endif()
endfunction()

ryty_add_library(ryty_gpu libSceVideoOut
    EXPORT_SOURCES
        core/libraries/gnmdriver/gnmdriver.cpp
        core/libraries/videoout/video_out.cpp
        core/libraries/ime/ime_dialog.cpp
        core/libraries/ime/ime.cpp
    SOURCES
        # ryGT: shadGT's on-screen keyboard dialog (libSceImeDialog), drawn with ImGui.
        src/core/libraries/ime/ime_dialog.cpp
        src/core/libraries/ime/ime_dialog_ui.cpp
        src/core/libraries/ime/ime.cpp
        src/core/libraries/ime/ime_ui.cpp
        src/core/libraries/ime/ime_kb_layout.cpp
        src/core/libraries/ime/ime_ui_shared.cpp
        ${SHADER_RECOMPILER}
        ${VIDEO_CORE}
        ${GNM_LIB}
        ${VIDEOOUT_LIB}
        src/imgui/renderer/imgui_core.cpp
        src/imgui/renderer/imgui_impl_sdl3.cpp
        src/imgui/renderer/imgui_impl_vulkan.cpp
        src/imgui/renderer/font_data.cpp
        src/imgui/renderer/font_stack.cpp
        src/imgui/renderer/texture_manager.cpp
        src/sdl_window.cpp
        ${INPUT}
        src/core/debug_state.cpp
        ryty_gpu/ryty_entry.cpp
        ryty_gpu/ryty_input_bridge.cpp
        ryty_gpu/ryty_write_watch.cpp
        # Stutter diagnostics (RYGT_DIAG=1), ryty_diag.h.
        ryty_gpu/ryty_diag.cpp
        # NVIDIA DLSS through the optional plugin dlss\ryGT_dlss.dll (ryty_dlss.h).
        ryty_gpu/ryty_dlss.cpp
        # AMD FSR 3.1 through amd_fidelityfx_vk.dll (ryty_fsr3.h).
        ryty_gpu/ryty_fsr3.cpp
        # AMD FSR 3.1 frame generation (ryty_framegen.h).
        ryty_gpu/ryty_framegen.cpp
        # NVIDIA DLSS frame generation through Streamline (ryty_dlssg.h).
        ryty_gpu/ryty_dlssg.cpp
        # FIX-074: shadGT's static red-zone patching, applied to the converted game code.
        ryty_gpu/ryty_red_zone.cpp
        ryty_gpu/ryty_game_patch.cpp
        src/core/cpu_patches.cpp
        src/core/loader/dwarf.cpp
)

target_compile_definitions(ryty_gpu PRIVATE RYTY_HAS_CPU_PATCHES=1 RYTY_DIAG=1)

# shadGT's Discord Rich Presence, in the renderer library only ("Running on ryGT").
if (ENABLE_DISCORD_RPC)
    target_sources(ryty_gpu PRIVATE src/common/discord_rpc_handler.cpp)
    target_compile_definitions(ryty_gpu PRIVATE ENABLE_DISCORD_RPC)
    target_link_libraries(ryty_gpu PRIVATE discord-rpc)
endif()

ryty_add_library(ryty_ngs2 libSceNgs2
    EXPORT_SOURCES
        core/libraries/ngs2/ngs2.cpp
    SOURCES
        src/core/libraries/ngs2/ngs2.cpp
        src/core/libraries/ngs2/ngs2_impl.cpp
        src/core/libraries/ngs2/ngs2_custom.cpp
        src/core/libraries/ngs2/ngs2_reverb.cpp
        src/core/libraries/ngs2/ngs2_geom.cpp
        src/core/libraries/ngs2/ngs2_pan.cpp
        src/core/libraries/ngs2/ngs2_report.cpp
        src/core/libraries/ngs2/ngs2_eq.cpp
        src/core/libraries/ngs2/ngs2_mastering.cpp
        src/core/libraries/ngs2/ngs2_sampler.cpp
        src/core/libraries/ngs2/ngs2_submixer.cpp
    HLE_HEADER core/libraries/ngs2/ngs2.h
    HLE_REGISTER Libraries::Ngs2::RegisterLib
    LOG shadgt_ngs2.log
)

# shadGT's audio decoder (ATRAC9, AAC, MP3), which the firmware's libSceNgs2 calls to decode
# compressed waveforms.
ryty_add_library(ryty_ajm libSceAjm
    EXPORT_SOURCES core/libraries/ajm/ajm.cpp
    SOURCES ${AJM_LIB}
    HLE_HEADER core/libraries/ajm/ajm.h
    HLE_REGISTER Libraries::Ajm::RegisterLib
    LOG shadgt_ajm.log
)
target_link_libraries(ryty_ajm PRIVATE LibAtrac9 fdk-aac FFmpeg::ffmpeg minimp3)

# shadGT's Audio3d (the SDL-backend version), mixing into Ryty's AudioOut ports.
ryty_add_library(ryty_audio3d libSceAudio3d
    EXPORT_SOURCES core/libraries/audio3d/audio3d.cpp
    SOURCES src/core/libraries/audio3d/audio3d.cpp ryty_gpu/ryty_audio3d_shims.cpp
    HLE_HEADER core/libraries/audio3d/audio3d.h
    HLE_REGISTER Libraries::Audio3d::RegisterLib
    LOG shadgt_audio3d.log
)

ryty_add_library(ryty_zlib libSceZlib
    EXPORT_SOURCES core/libraries/zlib/zlib.cpp
    SOURCES src/core/libraries/zlib/zlib.cpp
    HLE_HEADER core/libraries/zlib/zlib_sce.h
    HLE_REGISTER Libraries::Zlib::RegisterLib
    LOG shadgt_zlib.log
)

ryty_add_library(ryty_discmap libSceDiscMap
    EXPORT_SOURCES core/libraries/disc_map/disc_map.cpp
    SOURCES src/core/libraries/disc_map/disc_map.cpp
    HLE_HEADER core/libraries/disc_map/disc_map.h
    HLE_REGISTER Libraries::DiscMap::RegisterLib
    LOG shadgt_discmap.log
)

ryty_add_library(ryty_screenshot libSceScreenShot
    EXPORT_SOURCES core/libraries/screenshot/screenshot.cpp
    SOURCES src/core/libraries/screenshot/screenshot.cpp
    HLE_HEADER core/libraries/screenshot/screenshot.h
    HLE_REGISTER Libraries::ScreenShot::RegisterLib
    LOG shadgt_screenshot.log
)

# shadGT's NP libraries in one library; see ryty_gpu/ryty_np_entry.cpp.
set(RYTY_NP_SOURCES ${NP_LIBS})
list(FILTER RYTY_NP_SOURCES INCLUDE REGEX "/np/")
# The trophy pop-up is an ImGui overlay; ryty_np_shims.cpp logs unlocks instead.
list(FILTER RYTY_NP_SOURCES EXCLUDE REGEX "trophy_ui\.cpp$")
set(RYTY_NP_EXPORT_SOURCES)
foreach(source ${RYTY_NP_SOURCES})
    if (source MATCHES "\.cpp$")
        file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${source}" contents)
        if (contents MATCHES "LIB_FUNCTION")
            string(REGEX REPLACE "^src/" "" relative "${source}")
            list(APPEND RYTY_NP_EXPORT_SOURCES "${relative}")
        endif()
    endif()
endforeach()
ryty_add_library(ryty_np libSceNpAuth
    EXPORT_SOURCES ${RYTY_NP_EXPORT_SOURCES}
    SOURCES ${RYTY_NP_SOURCES} ${SHADNET} ryty_gpu/ryty_np_entry.cpp ryty_gpu/ryty_np_shims.cpp
            ryty_gpu/ryty_hle_stubs.cpp src/core/libraries/network/net_upnp.cpp
            src/core/libraries/network/net_util.cpp
)
target_compile_definitions(ryty_np PRIVATE RYTY_GUEST_PTHREADS)
target_link_libraries(ryty_np PRIVATE libprotobuf Cpp_Httplib minimp3 miniupnpc::miniupnpc ws2_32 iphlpapi)
target_include_directories(ryty_np PRIVATE "${SHADNET_PROTO_OUT}")

# Non-renderer libraries do not link the Vulkan shader compiler.
foreach(target IN ITEMS ryty_ngs2 ryty_ajm ryty_audio3d ryty_zlib ryty_discmap ryty_screenshot ryty_np)
    target_sources(${target} PRIVATE ryty_gpu/ryty_hle_precompile.cpp)
endforeach()
