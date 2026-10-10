// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/path_util.h"
#include "common/singleton.h"
#include "common/types.h"

#include <functional>
#include <thread>
#include <vector>

namespace Storage {

enum class BlobType : u32 {
    ShaderMeta,
    ShaderBinary,
    PipelineKey,
    ShaderProfile,
};

class DataBase {
public:
    static DataBase& Instance() {
        return *Common::Singleton<DataBase>::Instance();
    }

    void Open();
    void Close();
    [[nodiscard]] bool IsOpened() const {
        return opened;
    }
    void FinishPreload();

    bool Save(BlobType type, const std::string& name, std::vector<u8>&& data);
    bool Save(BlobType type, const std::string& name, std::vector<u32>&& data);

    void Load(BlobType type, const std::string& name, std::vector<u8>& data);
    void Load(BlobType type, const std::string& name, std::vector<u32>& data);
    /// Whether a blob is stored, without reporting a missing one (on-demand loading probes).
    [[nodiscard]] bool Exists(BlobType type, const std::string& name) const;

    void ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func);
    /// PERF-038: names (without extension) and modification times of the stored blobs of a
    /// type. Empty for an archived store.
    std::vector<std::pair<std::string, std::filesystem::file_time_type>> ListBlobs(
        BlobType type) const;

private:
    std::jthread io_worker{};
    std::filesystem::path cache_path{};
    bool opened{};
};

} // namespace Storage
