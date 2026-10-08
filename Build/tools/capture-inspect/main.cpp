#include <algorithm>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <spirv/unified1/spirv.hpp>
#include "renderdoc_replay.h"

using Json = nlohmann::json;
REPLAY_PROGRAM_MARKER()

static uint64_t Id(ResourceId id) {
    uint64_t value;
    static_assert(sizeof(value) == sizeof(id));
    std::memcpy(&value, &id, sizeof(value));
    return value;
}

static ResourceId Resource(uint64_t value) {
    ResourceId id;
    std::memcpy(&id, &value, sizeof(value));
    return id;
}


// Experiment: GCN min/max/clamp return the non-NaN operand; swap GLSL FMin/FMax/FClamp for
// the NaN-suppressing NMin/NMax/NClamp to test whether NaN handling explains an artifact.
static bytebuf NanSuppressingMinMax(const bytebuf& source, int& replaced) {
    std::vector<uint32_t> words(source.size() / 4);
    std::memcpy(words.data(), source.data(), source.size());
    uint32_t glsl = 0;
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> 16) {
        if ((words[offset] & 0xffff) == spv::OpExtInstImport &&
            std::strcmp(reinterpret_cast<const char*>(&words[offset + 2]), "GLSL.std.450") == 0)
            glsl = words[offset + 1];
    }
    replaced = 0;
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> 16) {
        if ((words[offset] & 0xffff) != spv::OpExtInst || words[offset + 3] != glsl) continue;
        auto& instruction = words[offset + 4];
        if (instruction == 37) { instruction = 79; ++replaced; }       // FMin -> NMin
        else if (instruction == 40) { instruction = 80; ++replaced; }  // FMax -> NMax
        else if (instruction == 43) { instruction = 81; ++replaced; }  // FClamp -> NClamp
    }
    bytebuf out;
    out.resize(words.size() * 4);
    std::memcpy(out.data(), words.data(), out.size());
    return out;
}

static bytebuf InterpolatedInputs(const bytebuf& source) {
    std::vector<uint32_t> words(source.size() / 4);
    std::memcpy(words.data(), source.data(), source.size());
    std::set<uint32_t> variables;
    std::map<uint32_t, uint32_t> pointer_elements;
    std::map<uint32_t, uint32_t> array_elements;
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> 16) {
        const auto op = spv::Op(words[offset] & 0xffff);
        if (op == spv::OpDecorate && words[offset + 2] == spv::DecorationPerVertexKHR)
            variables.insert(words[offset + 1]);
        if (op == spv::OpTypePointer && words[offset + 2] == spv::StorageClassInput)
            pointer_elements[words[offset + 1]] = words[offset + 3];
        if (op == spv::OpTypeArray) array_elements[words[offset + 1]] = words[offset + 2];
    }
    std::map<uint32_t, uint32_t> new_pointers;
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> 16) {
        if (spv::Op(words[offset] & 0xffff) == spv::OpVariable && variables.count(words[offset + 2])) {
            const auto pointer = words[offset + 1];
            new_pointers[pointer] = words[3]++;
        }
    }
    std::vector<uint32_t> result(words.begin(), words.begin() + 5);
    bool inserted_types = false;
    for (size_t offset = 5; offset < words.size(); offset += words[offset] >> 16) {
        const auto op = spv::Op(words[offset] & 0xffff);
        uint32_t count = words[offset] >> 16;
        if (op == spv::OpVariable && !inserted_types) {
            for (const auto& [pointer, replacement] : new_pointers) {
                result.insert(result.end(), {(4u << 16) | spv::OpTypePointer, replacement,
                    spv::StorageClassInput, array_elements.at(pointer_elements.at(pointer))});
            }
            inserted_types = true;
        }
        if (op == spv::OpDecorate && words[offset + 2] == spv::DecorationPerVertexKHR) continue;
        std::vector<uint32_t> instruction(words.begin() + offset, words.begin() + offset + count);
        if (op == spv::OpVariable && variables.count(instruction[2]))
            instruction[1] = new_pointers.at(instruction[1]);
        if (op == spv::OpAccessChain && variables.count(instruction[3])) {
            instruction.erase(instruction.begin() + 4);
            instruction[0] = (--count << 16) | spv::OpAccessChain;
        }
        result.insert(result.end(), instruction.begin(), instruction.end());
    }
    bytebuf bytes;
    bytes.resize(result.size() * 4);
    std::memcpy(bytes.data(), result.data(), bytes.size());
    return bytes;
}

static void Flatten(const rdcarray<ActionDescription>& actions,
                    std::vector<const ActionDescription*>& result) {
    for (const auto& action : actions) {
        result.push_back(&action);
        Flatten(action.children, result);
    }
}

static Json Pixel(const PixelValue& value) {
    return {value.floatValue[0], value.floatValue[1], value.floatValue[2], value.floatValue[3]};
}

static Json Variable(const ShaderVariable& variable) {
    Json result{{"name", variable.name.c_str()}, {"type", static_cast<int>(variable.type)}};
    result["float"] = Json::array();
    result["uint"] = Json::array();
    for (size_t i = 0; i < std::min<uint32_t>(16, variable.rows * variable.columns); ++i) {
        result["float"].push_back(variable.value.f32v[i]);
        result["uint"].push_back(variable.value.u32v[i]);
    }
    for (const auto& member : variable.members) result["members"].push_back(Variable(member));
    return result;
}

static void Write(const std::filesystem::path& path, const Json& value) {
    std::ofstream(path) << value.dump(2);
}

static Json Structured(const SDObject* object) {
    Json value{{"name", object->name.c_str()}, {"type", object->type.name.c_str()}};
    switch (object->type.basetype) {
    case SDBasic::Float: value["value"] = object->data.basic.d; break;
    case SDBasic::SignedInteger: value["value"] = object->data.basic.i; break;
    case SDBasic::Boolean: value["value"] = object->data.basic.b; break;
    default: value["value"] = object->data.basic.u; break;
    }
    for (size_t i = 0; i < object->NumChildren(); ++i)
        value["children"].push_back(Structured(object->GetChild(i)));
    return value;
}

static int Inspect(ICaptureFile* file, const std::filesystem::path& out, int argc, char** argv) {
    auto opened = file->OpenCapture(ReplayOptions{}, {});
    if (!opened.first.OK()) {
        std::cerr << "OpenCapture failed: " << static_cast<int>(opened.first.code);
        if (opened.first.internal_msg) std::cerr << ' ' << opened.first.internal_msg->c_str();
        std::cerr << '\n';
        return 3;
    }
    auto* replay = opened.second;
    if (argc > 4 && std::string(argv[3]) == "usage") {
        // Every event that reads or writes a resource, with RenderDoc's usage kind.
        for (int i = 4; i < argc; ++i) {
            for (const auto& use : replay->GetUsage(Resource(std::stoull(argv[i])))) {
                std::cout << argv[i] << ' ' << use.eventId << ' ' << static_cast<int>(use.usage)
                          << '\n';
            }
        }
        replay->Shutdown();
        return 0;
    }
    std::vector<const ActionDescription*> actions;
    Flatten(replay->GetRootActions(), actions);
    Json summary;
    std::map<uint64_t, std::string> names;
    for (const auto& resource : replay->GetResources()) {
        names[Id(resource.resourceId)] = resource.name.c_str();
        summary["resources"].push_back({{"id", Id(resource.resourceId)},
                                         {"name", resource.name.c_str()},
                                         {"type", static_cast<int>(resource.type)}});
    }
    uint32_t final_event = 0;
    ResourceId final_target;
    std::set<uint64_t> outputs;
    for (const auto* action : actions) {
        Json targets = Json::array();
        for (ResourceId target : action->outputs) {
            targets.push_back(Id(target));
            if (target != ResourceId()) outputs.insert(Id(target));
        }
        summary["actions"].push_back({{"event", action->eventId},
                                       {"name", action->customName.c_str()},
                                       {"flags", static_cast<uint32_t>(action->flags)},
                                       {"vertices", action->numIndices},
                                       {"instances", action->numInstances},
                                       {"outputs", targets},
                                       {"source", Id(action->copySource)},
                                       {"destination", Id(action->copyDestination)}});
        final_event = std::max(final_event, action->eventId);
        if ((action->flags & ActionFlags::Drawcall) && action->outputs[0] != ResourceId()) {
            final_target = action->outputs[0];
        }
    }
    summary["final_event"] = final_event;
    summary["final_draw_target"] = Id(final_target);
    for (const auto& texture : replay->GetTextures()) {
        summary["textures"].push_back({{"id", Id(texture.resourceId)},
                                        {"name", names[Id(texture.resourceId)]},
                                        {"width", texture.width}, {"height", texture.height},
                                        {"depth", texture.depth}, {"layers", texture.arraysize},
                                        {"mips", texture.mips}, {"samples", texture.msSamp},
                                        {"format", texture.format.Name().c_str()},
                                        {"flags", static_cast<uint32_t>(texture.creationFlags)}});
    }
    Write(out / "summary.json", summary);
    std::cout << actions.size() << " actions, " << replay->GetTextures().size()
              << " textures; final target " << Id(final_target) << '\n';

    if (argc > 4 && std::string(argv[3]) == "ib-dump") {
        // The draw's real index buffer and primitive restart state.
        const uint32_t event = std::stoul(argv[4]);
        replay->SetFrameEvent(event, false);
        const auto* vk = replay->GetVulkanPipelineState();
        const auto& ia = vk->inputAssembly;
        const ActionDescription* action = nullptr;
        for (const auto* a : actions) if (a->eventId == event) action = a;
        std::ofstream dump(out / "ib-dump.txt");
        dump << "restart=" << ia.primitiveRestartEnable << " stride=" << ia.indexBuffer.byteStride
             << " offset=" << ia.indexBuffer.byteOffset << " count=" << action->numIndices
             << " first=" << action->indexOffset << " base=" << action->baseVertex << '\n';
        const uint32_t stride = ia.indexBuffer.byteStride ? ia.indexBuffer.byteStride : 4;
        const bytebuf data = replay->GetBufferData(
            ia.indexBuffer.resourceId, ia.indexBuffer.byteOffset + uint64_t(action->indexOffset) * stride,
            uint64_t(action->numIndices) * stride);
        for (size_t i = 0; i * stride < data.size(); ++i) {
            uint32_t index = 0;
            std::memcpy(&index, data.data() + i * stride, stride);
            dump << index << (i % 4 == 3 ? '\n' : ' ');
        }
        replay->Shutdown();
        return 0;
    }

    if (argc > 7 && std::string(argv[3]) == "buffer-dump") {
        // Raw dwords of a buffer at an event: buffer-dump <event> <resource> <offset> <bytes>.
        replay->SetFrameEvent(std::stoul(argv[4]), false);
        const bytebuf data = replay->GetBufferData(Resource(std::stoull(argv[5])),
                                                   std::stoull(argv[6]), std::stoull(argv[7]));
        std::ofstream dump(out / "buffer-dump.txt");
        for (size_t offset = 0; offset + 4 <= data.size(); offset += 4) {
            uint32_t word;
            std::memcpy(&word, data.data() + offset, 4);
            dump << word << (offset % 56 == 52 ? '\n' : ' ');
        }
        replay->Shutdown();
        return 0;
    }

    if (argc > 5 && std::string(argv[3]) == "mesh-dump") {
        // Clip-space positions of one draw's vertex shader (vs) or tessellation (gs) output.
        replay->SetFrameEvent(std::stoul(argv[4]), false);
        const auto stage = std::string(argv[5]) == "gs" ? MeshDataStage::GSOut : MeshDataStage::VSOut;
        const MeshFormat mesh = replay->GetPostVSData(0, 0, stage);
        const bytebuf data =
            replay->GetBufferData(mesh.vertexResourceId, mesh.vertexByteOffset, 64 << 20);
        std::ofstream dump(out / "mesh-dump.txt");
        dump << "stride=" << mesh.vertexByteStride << " topology=" << static_cast<int>(mesh.topology)
             << " indices=" << mesh.numIndices << " indexed=" << (mesh.indexResourceId != ResourceId())
             << '\n';
        if (mesh.indexResourceId != ResourceId()) {
            const bytebuf indices = replay->GetBufferData(
                mesh.indexResourceId, mesh.indexByteOffset, mesh.numIndices * mesh.indexByteStride);
            std::ofstream index_dump(out / "index-dump.txt");
            index_dump << "base_vertex=" << mesh.baseVertex << " stride=" << mesh.indexByteStride << '\n';
            for (uint32_t i = 0; i < mesh.numIndices; ++i) {
                uint32_t index = 0;
                std::memcpy(&index, indices.data() + i * mesh.indexByteStride, mesh.indexByteStride);
                index_dump << index << (i % 4 == 3 ? '\n' : ' ');
            }
        }
        const size_t count = mesh.vertexByteStride ? data.size() / mesh.vertexByteStride : 0;
        for (size_t v = 0; v < count && v < 200000; ++v) {
            float p[4];
            std::memcpy(p, data.data() + v * mesh.vertexByteStride, sizeof(p));
            dump << v << ' ' << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << p[3] << '\n';
        }
        replay->Shutdown();
        return 0;
    }

    if (argc > 3 && std::string(argv[3]) == "sliver-scan") {
        // Stretched geometry: triangles (tessellation output, a triangle list) whose longest
        // screen-space edge is long while their area is tiny.
        std::ofstream report(out / "sliver-scan.txt");
        for (const auto* action : actions) {
            if (!(action->flags & ActionFlags::Drawcall)) continue;
            replay->SetFrameEvent(action->eventId, false);
            const MeshFormat mesh = replay->GetPostVSData(0, 0, MeshDataStage::GSOut);
            if (mesh.vertexResourceId == ResourceId() || mesh.vertexByteStride < 16) continue;
            const bytebuf data =
                replay->GetBufferData(mesh.vertexResourceId, mesh.vertexByteOffset, 64 << 20);
            const size_t count = data.size() / mesh.vertexByteStride;
            size_t slivers = 0, triangles = 0;
            float longest = 0.f;
            float example[6]{};
            for (size_t t = 0; t + 2 < count; t += 3) {
                float sx[3], sy[3];
                bool visible = true;
                for (int k = 0; k < 3; ++k) {
                    float p[4];
                    std::memcpy(p, data.data() + (t + k) * mesh.vertexByteStride, sizeof(p));
                    if (!(p[3] > 1e-4f) || !std::isfinite(p[0]) || !std::isfinite(p[1])) {
                        visible = false;
                        break;
                    }
                    sx[k] = (p[0] / p[3] * 0.5f + 0.5f) * 1920.f;
                    sy[k] = (p[1] / p[3] * 0.5f + 0.5f) * 1080.f;
                }
                if (!visible) continue;
                ++triangles;
                float edge = 0.f;
                for (int k = 0; k < 3; ++k) {
                    const float dx = sx[k] - sx[(k + 1) % 3], dy = sy[k] - sy[(k + 1) % 3];
                    edge = std::max(edge, std::sqrt(dx * dx + dy * dy));
                }
                const float area = std::fabs((sx[1] - sx[0]) * (sy[2] - sy[0]) -
                                             (sx[2] - sx[0]) * (sy[1] - sy[0])) * 0.5f;
                const bool on_screen = std::max({sx[0], sx[1], sx[2]}) > 0 &&
                                       std::min({sx[0], sx[1], sx[2]}) < 1920 &&
                                       std::max({sy[0], sy[1], sy[2]}) > 0 &&
                                       std::min({sy[0], sy[1], sy[2]}) < 1080;
                if (on_screen && edge > 300.f && area < edge * 4.f) {
                    if (edge > longest) {
                        longest = edge;
                        for (int k = 0; k < 3; ++k) { example[k * 2] = sx[k]; example[k * 2 + 1] = sy[k]; }
                    }
                    ++slivers;
                }
            }
            if (slivers > 0) {
                report << action->eventId << " triangles=" << triangles << " slivers=" << slivers
                       << " longest=" << longest << " example=(" << example[0] << ',' << example[1]
                       << ")(" << example[2] << ',' << example[3] << ")(" << example[4] << ','
                       << example[5] << ") indices=" << action->numIndices << '\n';
                report.flush();
            }
        }
        replay->Shutdown();
        return 0;
    }

    if (argc > 3 && std::string(argv[3]) == "vsout-scan") {
        // Exploded geometry: per draw, how many transformed positions are non-finite or
        // absurdly far outside the clip volume.
        const uint32_t first = argc > 4 ? std::stoul(argv[4]) : 0;
        const uint32_t last = argc > 5 ? std::stoul(argv[5]) : UINT32_MAX;
        std::ofstream report(out / "vsout-scan.txt");
        for (const auto* action : actions) {
            if (!(action->flags & ActionFlags::Drawcall) || action->eventId < first ||
                action->eventId > last) {
                continue;
            }
            replay->SetFrameEvent(action->eventId, false);
            const bool vs_only = argc > 6 && std::string(argv[6]) == "vs";
            for (const auto stage : {MeshDataStage::GSOut, MeshDataStage::VSOut}) {
                if (vs_only && stage == MeshDataStage::GSOut) continue;
                const MeshFormat mesh = replay->GetPostVSData(0, 0, stage);
                if (mesh.vertexResourceId == ResourceId() || mesh.vertexByteStride < 16) {
                    continue;
                }
                const bytebuf data = replay->GetBufferData(mesh.vertexResourceId,
                                                           mesh.vertexByteOffset, 64 << 20);
                const size_t count = data.size() / mesh.vertexByteStride;
                size_t bad = 0, far = 0;
                float worst = 0.f;
                for (size_t v = 0; v < count; ++v) {
                    float p[4];
                    std::memcpy(p, data.data() + v * mesh.vertexByteStride, sizeof(p));
                    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) ||
                        !std::isfinite(p[3])) {
                        ++bad;
                        continue;
                    }
                    const float w = std::max(std::fabs(p[3]), 1e-6f);
                    const float extent = std::max(std::fabs(p[0]), std::fabs(p[1])) / w;
                    worst = std::max(worst, extent);
                    if (extent > 1000.f || std::fabs(p[3]) > 1e7f) {
                        ++far;
                    }
                }
                report << action->eventId << ' ' << (stage == MeshDataStage::GSOut ? "gs" : "vs")
                       << " verts=" << count << " nonfinite=" << bad << " far=" << far
                       << " worst=" << worst << " indices=" << action->numIndices
                       << " name=" << action->customName.c_str() << '\n';
                break;
            }
        }
        replay->Shutdown();
        return 0;
    }

    const uint32_t event = argc > 3 ? std::stoul(argv[3]) : final_event;
    replay->SetFrameEvent(event, true);
    const auto& structured = replay->GetStructuredFile();
    for (const auto* action : actions) {
        if (action->eventId != event) continue;
        Json chunks = Json::array();
        for (const auto& api : action->events) {
            if (api.chunkIndex < structured.chunks.size())
                chunks.push_back(Structured(structured.chunks[api.chunkIndex]));
        }
        Write(out / "event-chunks.json", chunks);
        if (argc > 7 && (std::string(argv[7]) == "clear-stencil" || std::string(argv[7]) == "unswizzle-clear" || std::string(argv[7]) == "far-depth-range")) {
            const bool stencil_mode = std::string(argv[7]) == "clear-stencil";
            const bool far_depth_mode = std::string(argv[7]) == "far-depth-range";
            const auto& full = file->GetStructuredData();
            SDFile modified;
            modified.version = full.version;
            for (auto* buffer : full.buffers) modified.buffers.push_back(new bytebuf(*buffer));
            for (const auto* chunk : full.chunks) modified.chunks.push_back(chunk->Duplicate());
            bool changed = false;
            if (far_depth_mode) {
                for (auto* chunk : modified.chunks) {
                    if (chunk->name != "vkCmdSetViewport" && chunk->name != "vkCmdSetViewportWithCount" && chunk->name != "vkCmdSetViewportWithCountEXT") continue;
                    auto* viewports = chunk->FindChildRecursively("pViewports");
                    if (!viewports) continue;
                    for (size_t i = 0; i < viewports->NumChildren(); ++i) {
                        auto* viewport = viewports->GetChild(i);
                        auto* minimum = viewport->FindChild("minDepth");
                        auto* maximum = viewport->FindChild("maxDepth");
                        if (minimum && maximum && minimum->data.basic.d >= 1.0 - std::ldexp(1.0, -20) && minimum->data.basic.d < 1.0 && maximum->data.basic.d == 1.0) {
                            maximum->data.basic.d = std::nextafter(1.f, 0.f);
                            changed = true;
                            std::cout << "Adjusted far viewport depth range\n";
                        }
                    }
                }
            }
            for (const auto& api : action->events) {
                if (far_depth_mode) continue;
                if (api.chunkIndex >= modified.chunks.size()) continue;
                auto* chunk = modified.chunks[api.chunkIndex];
                if (chunk->name != "vkCmdBeginRendering" && chunk->name != "vkCmdBeginRenderingKHR") continue;
                if (stencil_mode) {
                    auto* stencil = chunk->FindChildRecursively("pStencilAttachment");
                    if (!stencil) continue;
                    auto* load = stencil->FindChildRecursively("loadOp");
                    auto* clear = stencil->FindChildRecursively("stencil");
                    if (!load || !clear) continue;
                    load->data.basic.u = 1; // VK_ATTACHMENT_LOAD_OP_CLEAR
                    clear->data.basic.u = 0;
                    changed = true;
                } else {
                    auto* colors = chunk->FindChildRecursively("pColorAttachments");
                    if (!colors) continue;
                    for (size_t i = 0; i < colors->NumChildren(); ++i) {
                        auto* attachment = colors->GetChild(i);
                        auto* load = attachment->FindChildRecursively("loadOp");
                        auto* clear = attachment->FindChildRecursively("clearValue");
                        if (!load || load->data.basic.u != 1 || !clear) continue;
                        auto* color = clear->FindChild("color");
                        auto* words = color->FindChild("uint32");
                        uint32_t reversed[4];
                        for (size_t c = 0; c < 4; ++c)
                            reversed[c] = words->GetChild(3 - c)->data.basic.u;
                        for (size_t view = 0; view < color->NumChildren(); ++view) {
                            auto* values = color->GetChild(view);
                            for (size_t c = 0; c < 4; ++c) {
                                auto* component = values->GetChild(c);
                                if (component->type.basetype == SDBasic::Float) {
                                    float value;
                                    std::memcpy(&value, &reversed[c], 4);
                                    component->data.basic.d = value;
                                } else {
                                    component->data.basic.u = reversed[c];
                                }
                            }
                        }
                        float depth;
                        std::memcpy(&depth, &reversed[0], 4);
                        clear->FindChild("depthStencil")->FindChild("depth")->data.basic.d = depth;
                        clear->FindChild("depthStencil")->FindChild("stencil")->data.basic.u = reversed[1];
                        changed = true;
                    }
                }
            }
            if (changed) {
                auto* destination = RENDERDOC_OpenCaptureFile();
                const auto thumbnail = file->GetThumbnail(FileType::PNG, 0);
                destination->SetMetadata(file->DriverName(), 0, FileType::PNG, thumbnail.width,
                    thumbnail.height, thumbnail.data, file->TimestampBase(), file->TimestampFrequency());
                const auto filename = far_depth_mode ? "far-depth-range.rdc" : stencil_mode ? "stencil-cleared.rdc" : "clear-unswizzled.rdc";
                auto result = destination->Convert((out / filename).string().c_str(), "rdc", &modified, {});
                std::cout << "Stencil-clear conversion: " << result.OK() << '\n';
                destination->Shutdown();
            } else {
                std::cout << "No stencil attachment changed\n";
            }
            replay->Shutdown();
            return changed ? 0 : 4;
        }
    }
    if (argc <= 3) {
        for (const auto& texture : replay->GetTextures()) {
            if (!outputs.count(Id(texture.resourceId))) continue;
            TextureSave save;
            save.resourceId = texture.resourceId;
            save.destType = FileType::PNG;
            save.mip = 0;
            auto path = out / ("target-" + std::to_string(Id(texture.resourceId)) + ".png");
            auto result = replay->SaveTexture(save, path.string().c_str());
            std::cout << "Saved " << path.filename().string() << ": " << result.OK() << '\n';
        }
    }
    if (argc > 4) {
        ResourceId target = Resource(std::stoull(argv[4]));
        TextureSave save;
        save.resourceId = target;
        save.destType = FileType::PNG;
        save.mip = 0;
        replay->SaveTexture(save, (out / "selected.png").string().c_str());
        // Raw float contents for numeric comparison (PNG clamps to 8 bits).
        TextureSave raw = save;
        raw.destType = FileType::DDS;
        replay->SaveTexture(raw, (out / "selected.dds").string().c_str());
        if (argc > 6) {
            auto history = replay->PixelHistory(target, std::stoul(argv[5]), std::stoul(argv[6]),
                                                Subresource{}, CompType::Typeless);
            Json records = Json::array();
            for (const auto& change : history) {
                records.push_back({{"event", change.eventId}, {"fragment", change.fragIndex},
                                   {"before", Pixel(change.preMod.col)},
                                   {"shader", Pixel(change.shaderOut.col)},
                                   {"after", Pixel(change.postMod.col)},
                                   {"stencil_before", change.preMod.stencil},
                                   {"stencil_after", change.postMod.stencil},
                                   {"depth_before", change.preMod.depth},
                                   {"shader_depth", change.shaderOut.depth},
                                   {"depth_after", change.postMod.depth},
                                   {"passed", change.Passed()},
                                   {"depth_clipped", change.depthClipped},
                                   {"bounds_failed", change.depthBoundsFailed},
                                   {"depth_failed", change.depthTestFailed},
                                   {"stencil_failed", change.stencilTestFailed},
                                   {"discarded", change.shaderDiscarded},
                                   {"sample_masked", change.sampleMasked},
                                   {"culled", change.backfaceCulled},
                                   {"direct_write", change.directShaderWrite}});
            }
            Write(out / "pixel-history.json", records);
            replay->SetFrameEvent(event, true);
        }
    }
    if (argc > 3) {
        Json state;
        const auto* vk = replay->GetVulkanPipelineState();
        state["event"] = event;
        state["fragment_shader"] = Id(vk->fragmentShader.resourceId);
        state["fragment_name"] = names[Id(vk->fragmentShader.resourceId)];
        state["vertex_shader"] = Id(vk->vertexShader.resourceId);
        state["vertex_name"] = names[Id(vk->vertexShader.resourceId)];
        state["depth_test"] = vk->depthStencil.depthTestEnable;
        state["depth_write"] = vk->depthStencil.depthWriteEnable;
        state["depth_function"] = static_cast<int>(vk->depthStencil.depthFunction);
        state["stencil_test"] = vk->depthStencil.stencilTestEnable;
        state["stencil_reference"] = vk->depthStencil.frontFace.reference;
        state["stencil_function"] = static_cast<int>(vk->depthStencil.frontFace.function);
        for (const auto* face : {&vk->depthStencil.frontFace, &vk->depthStencil.backFace}) {
            state["stencil_faces"].push_back({{"reference", face->reference},
                {"compare_mask", face->compareMask}, {"write_mask", face->writeMask},
                {"function", static_cast<int>(face->function)},
                {"fail", static_cast<int>(face->failOperation)},
                {"pass", static_cast<int>(face->passOperation)},
                {"depth_fail", static_cast<int>(face->depthFailOperation)}});
        }
        for (const auto& attachment : vk->currentPass.framebuffer.attachments) {
            state["attachments"].push_back({{"resource", Id(attachment.resource)},
                {"name", names[Id(attachment.resource)]},
                {"format", attachment.format.Name().c_str()}});
        }
        state["color_feedback"] = vk->currentPass.colorFeedbackAllowed;
        state["negative_one_to_one"] = vk->viewportScissor.depthNegativeOneToOne;
        state["depth_clip"] = vk->rasterizer.depthClipEnable;
        state["depth_clamp"] = vk->rasterizer.depthClampEnable;
        state["cull_mode"] = static_cast<int>(vk->rasterizer.cullMode);
        state["front_ccw"] = vk->rasterizer.frontCCW;
        for (const auto& blend : vk->colorBlend.blends) {
            state["blends"].push_back({{"enabled", blend.enabled}, {"write_mask", blend.writeMask},
                {"color_src", static_cast<int>(blend.colorBlend.source)},
                {"color_dst", static_cast<int>(blend.colorBlend.destination)},
                {"color_op", static_cast<int>(blend.colorBlend.operation)},
                {"alpha_src", static_cast<int>(blend.alphaBlend.source)},
                {"alpha_dst", static_cast<int>(blend.alphaBlend.destination)},
                {"alpha_op", static_cast<int>(blend.alphaBlend.operation)}});
        }
        for (size_t i = 0; i + 4 <= vk->pushconsts.size(); i += 4) {
            uint32_t value;
            std::memcpy(&value, vk->pushconsts.data() + i, 4);
            state["push_words"].push_back(value);
        }
        for (const auto& viewport : vk->viewportScissor.viewportScissors) {
            state["viewports"].push_back({{"x", viewport.vp.x}, {"y", viewport.vp.y},
                                           {"width", viewport.vp.width},
                                           {"height", viewport.vp.height},
                                           {"min_depth", viewport.vp.minDepth},
                                           {"max_depth", viewport.vp.maxDepth},
                                           {"scissor_x", viewport.scissor.x},
                                           {"scissor_y", viewport.scissor.y},
                                           {"scissor_width", viewport.scissor.width},
                                           {"scissor_height", viewport.scissor.height}});
        }
        for (const auto& attribute : vk->vertexInput.attributes) {
            state["vertex_attributes"].push_back(
                {{"location", attribute.location}, {"binding", attribute.binding},
                 {"format", attribute.format.Name().c_str()}, {"offset", attribute.byteOffset}});
        }
        for (size_t i = 0; i < vk->vertexInput.vertexBuffers.size(); ++i) {
            const auto& buffer = vk->vertexInput.vertexBuffers[i];
            state["vertex_buffers"].push_back(
                {{"binding", i}, {"resource", Id(buffer.resourceId)}, {"offset", buffer.byteOffset},
                 {"stride", buffer.byteStride}, {"size", buffer.byteSize}});
            if (buffer.resourceId != ResourceId()) {
                const auto bytes = replay->GetBufferData(buffer.resourceId, buffer.byteOffset,
                                                          std::min<uint64_t>(buffer.byteSize, 4096));
                Json words = Json::array();
                for (size_t offset = 0; offset + 4 <= bytes.size(); offset += 4) {
                    uint32_t word;
                    std::memcpy(&word, bytes.data() + offset, 4);
                    words.push_back(word);
                }
                state["vertex_buffers"].back()["words"] = words;
            }
        }
        const auto& accesses = replay->GetDescriptorAccess();
        for (const auto& access : accesses) {
            auto descriptors = replay->GetDescriptors(access.descriptorStore,
                                                      {DescriptorRange(access)});
            if (descriptors.empty()) continue;
            const auto& descriptor = descriptors[0];
            state["descriptors"].push_back({{"stage", static_cast<int>(access.stage)},
                                             {"index", access.index},
                                             {"type", static_cast<int>(access.type)},
                                             {"resource", Id(descriptor.resource)},
                                             {"name", names[Id(descriptor.resource)]},
                                             {"format", descriptor.format.Name().c_str()},
                                             {"offset", descriptor.byteOffset},
                                             {"size", descriptor.byteSize}});
            state["descriptors"].back()["swizzle"] = {
                static_cast<int>(descriptor.swizzle.red), static_cast<int>(descriptor.swizzle.green),
                static_cast<int>(descriptor.swizzle.blue), static_cast<int>(descriptor.swizzle.alpha)};
            if ((access.type == DescriptorType::ConstantBuffer || access.type == DescriptorType::Buffer ||
                 access.type == DescriptorType::ReadWriteBuffer) &&
                descriptor.resource != ResourceId()) {
                const auto bytes = replay->GetBufferData(descriptor.resource, descriptor.byteOffset,
                                                        std::min<uint64_t>(descriptor.byteSize, 256));
                Json words = Json::array();
                Json floats = Json::array();
                for (size_t i = 0; i + 4 <= bytes.size(); i += 4) {
                    uint32_t word;
                    float value;
                    std::memcpy(&word, bytes.data() + i, 4);
                    std::memcpy(&value, bytes.data() + i, 4);
                    words.push_back(word);
                    floats.push_back(value);
                }
                state["descriptors"].back()["words"] = words;
                state["descriptors"].back()["floats"] = floats;
            }
            if (access.stage == ShaderStage::Fragment && descriptor.resource != ResourceId()) {
                auto found = std::find_if(replay->GetTextures().begin(), replay->GetTextures().end(),
                    [&](const auto& texture) { return texture.resourceId == descriptor.resource; });
                if (found != replay->GetTextures().end()) {
                    TextureSave save;
                    save.resourceId = descriptor.resource;
                    save.destType = FileType::PNG;
                    save.mip = 0;
                    const auto path = out / ("input-" + std::to_string(access.index) + ".png");
                    replay->SaveTexture(save, path.string().c_str());
                }
            }
        }
        if (vk->fragmentShader.reflection) {
            const auto* reflection = vk->fragmentShader.reflection;
            const auto disassembly = replay->DisassembleShader(vk->graphics.pipelineResourceId, reflection, "");
            std::ofstream(out / "fragment.txt") << disassembly.c_str();
            std::ofstream binary(out / "fragment.spv", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(reflection->rawBytes.data()),
                         reflection->rawBytes.size());
        }
        if (vk->vertexShader.reflection) {
            const auto* reflection = vk->vertexShader.reflection;
            const auto disassembly = replay->DisassembleShader(vk->graphics.pipelineResourceId, reflection, "");
            std::ofstream(out / "vertex.txt") << disassembly.c_str();
            std::ofstream binary(out / "vertex.spv", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(reflection->rawBytes.data()),
                         reflection->rawBytes.size());
            const auto input_mesh = replay->GetPostVSData(0, 0, MeshDataStage::VSIn);
            state["input_mesh"] = {{"resource", Id(input_mesh.vertexResourceId)},
                                    {"stride", input_mesh.vertexByteStride},
                                    {"vertices", input_mesh.numIndices}};
            if (input_mesh.vertexResourceId != ResourceId() && input_mesh.vertexByteStride) {
                const auto bytes = replay->GetBufferData(
                    input_mesh.vertexResourceId, input_mesh.vertexByteOffset,
                    uint64_t(std::min<uint32_t>(input_mesh.numIndices, 4096)) *
                        input_mesh.vertexByteStride);
                for (size_t row = 0; row + input_mesh.vertexByteStride <= bytes.size();
                     row += input_mesh.vertexByteStride) {
                    Json values = Json::array();
                    for (size_t offset = 0; offset + 4 <= input_mesh.vertexByteStride; offset += 4) {
                        float value;
                        std::memcpy(&value, bytes.data() + row + offset, 4);
                        values.push_back(value);
                    }
                    state["input_mesh"]["data"].push_back(values);
                }
            }
            const auto mesh = replay->GetPostVSData(0, 0, MeshDataStage::VSOut);
            state["mesh"] = {{"resource", Id(mesh.vertexResourceId)},
                              {"stride", mesh.vertexByteStride}, {"vertices", mesh.numIndices},
                              {"topology", static_cast<int>(mesh.topology)},
                              {"index_resource", Id(mesh.indexResourceId)},
                              {"index_stride", mesh.indexByteStride},
                              {"base_vertex", mesh.baseVertex}};
            if (mesh.indexResourceId != ResourceId() && mesh.indexByteStride) {
                const auto count = std::min<uint32_t>(mesh.numIndices, 4096);
                const auto indices = replay->GetBufferData(mesh.indexResourceId, mesh.indexByteOffset,
                    uint64_t(count) * mesh.indexByteStride);
                for (size_t offset = 0; offset + mesh.indexByteStride <= indices.size();
                     offset += mesh.indexByteStride) {
                    uint32_t value = 0;
                    std::memcpy(&value, indices.data() + offset, mesh.indexByteStride);
                    state["mesh"]["indices"].push_back(value);
                }
            }
            if (mesh.vertexResourceId != ResourceId() && mesh.vertexByteStride) {
                const auto bytes = replay->GetBufferData(mesh.vertexResourceId, mesh.vertexByteOffset,
                    uint64_t(std::min<uint32_t>(mesh.numIndices, 4096)) * mesh.vertexByteStride);
                Json rows = Json::array();
                for (size_t row = 0; row + mesh.vertexByteStride <= bytes.size(); row += mesh.vertexByteStride) {
                    Json values = Json::array();
                    for (size_t offset = 0; offset + 4 <= mesh.vertexByteStride; offset += 4) {
                        float value;
                        std::memcpy(&value, bytes.data() + row + offset, 4);
                        values.push_back(value);
                    }
                    rows.push_back(values);
                }
                state["mesh"]["data"] = rows;
            }
        }
        Write(out / "state.json", state);
        if (argc > 7 && std::string(argv[7]) == "nan-minmax") {
            const auto original = vk->fragmentShader.resourceId;
            int replaced = 0;
            const auto modified = NanSuppressingMinMax(vk->fragmentShader.reflection->rawBytes, replaced);
            const auto shader = replay->BuildTargetShader("main", ShaderEncoding::SPIRV, modified,
                                                          ShaderCompileFlags{}, ShaderStage::Fragment);
            std::cout << "Replaced " << replaced << " min/max/clamp; shader " << Id(shader.first) << ' '
                      << shader.second.c_str() << std::endl;
            if (shader.first != ResourceId()) {
                replay->ReplaceResource(original, shader.first);
                replay->SetFrameEvent(event, true);
                TextureSave save;
                save.resourceId = Resource(std::stoull(argv[4]));
                save.destType = FileType::DDS;
                save.mip = 0;
                replay->SaveTexture(save, (out / "nan-minmax-target.dds").string().c_str());
                save.destType = FileType::PNG;
                replay->SaveTexture(save, (out / "nan-minmax-target.png").string().c_str());
                replay->SetFrameEvent(final_event, true);
                save.resourceId = final_target;
                replay->SaveTexture(save, (out / "nan-minmax-frame.png").string().c_str());
                replay->RemoveReplacement(original);
                replay->FreeTargetResource(shader.first);
            }
        } else if (argc > 7 && std::string(argv[7]) == "interpolate") {
            const auto original = vk->fragmentShader.resourceId;
            const auto modified = InterpolatedInputs(vk->fragmentShader.reflection->rawBytes);
            std::ofstream binary(out / "interpolated.spv", std::ios::binary);
            binary.write(reinterpret_cast<const char*>(modified.data()), modified.size());
            binary.close();
            const auto shader = replay->BuildTargetShader("main", ShaderEncoding::SPIRV, modified,
                                                          ShaderCompileFlags{}, ShaderStage::Fragment);
            std::cout << "Replacement shader: " << Id(shader.first) << ' ' << shader.second.c_str() << '\n';
            if (shader.first != ResourceId()) {
                replay->ReplaceResource(original, shader.first);
                replay->SetFrameEvent(final_event, true);
                TextureSave save;
                save.resourceId = final_target;
                save.destType = FileType::PNG;
                save.mip = 0;
                replay->SaveTexture(save, (out / "interpolated-frame.png").string().c_str());
                replay->RemoveReplacement(original);
                replay->FreeTargetResource(shader.first);
            }
        } else if (argc > 7) {
            auto* trace = replay->DebugPixel(std::stoul(argv[5]), std::stoul(argv[6]), DebugPixelInputs{});
            Json debug;
            debug["valid"] = trace && trace->debugger;
            if (trace) {
                for (const auto& variable : trace->inputs) debug["inputs"].push_back(Variable(variable));
                for (const auto& variable : trace->constantBlocks) debug["constants"].push_back(Variable(variable));
                if (trace->debugger) {
                    for (size_t i = 0; i < 10000; ++i) {
                        auto steps = replay->ContinueDebug(trace->debugger);
                        if (steps.empty()) break;
                        for (const auto& step : steps) {
                            Json record{{"instruction", step.nextInstruction}, {"step", step.stepIndex}};
                            for (const auto& change : step.changes) record["changes"].push_back(Variable(change.after));
                            debug["steps"].push_back(record);
                        }
                    }
                }
                replay->FreeTrace(trace);
            }
            Write(out / "debug.json", debug);
        }
    }
    replay->Shutdown();
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "capture-inspect capture.rdc output-dir [event [texture [x y]]]\n";
        return 1;
    }
    std::filesystem::path out(argv[2]);
    std::filesystem::create_directories(out);
    GlobalEnvironment environment;
    environment.enumerateGPUs = false;
    RENDERDOC_InitialiseReplay(environment, {});
    auto* file = RENDERDOC_OpenCaptureFile();
    auto result = file->OpenFile(argv[1], "", {});
    int status = 2;
    if (result.OK() && argc > 3 && std::string(argv[3]) == "thumb") {
        // Embedded capture thumbnail only; no replay.
        const auto thumbnail = file->GetThumbnail(FileType::PNG, 0);
        std::ofstream(out / "thumb.png", std::ios::binary)
            .write(reinterpret_cast<const char*>(thumbnail.data.data()), thumbnail.data.size());
        std::cout << "Thumbnail " << thumbnail.width << 'x' << thumbnail.height << std::endl;
        status = 0;
    } else if (result.OK()) status = Inspect(file, out, argc, argv);
    else std::cerr << "OpenFile failed: " << static_cast<int>(result.code) << '\n';
    file->Shutdown();
    RENDERDOC_ShutdownReplay();
    return status;
}
