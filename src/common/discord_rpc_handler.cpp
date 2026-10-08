// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "discord_rpc_handler.h"

namespace DiscordRPCHandler {

void RPC::init() {
    DiscordEventHandlers handlers{};
    Discord_Initialize("1557422166931275906", &handlers, 1, nullptr);
    rpcEnabled = true;
}

void RPC::setStatusIdling() {
    DiscordRichPresence rpc{};
    rpc.details = "Running on shadGT";
    rpc.largeImageKey = "https://raw.githubusercontent.com/mlgprorektm8/shadPS4/main/assets/"
                        "GranTurismoSportIcon.jpg";
    rpc.largeImageText = "Gran Turismo Sport";
    status = RPCStatus::Idling;
    Discord_UpdatePresence(&rpc);
}

void RPC::setStatusPlaying(const std::string&, const std::string&) {
    DiscordRichPresence rpc{};
    rpc.details = "Running on shadGT";
    rpc.largeImageKey = "https://raw.githubusercontent.com/mlgprorektm8/shadPS4/main/assets/"
                        "GranTurismoSportIcon.jpg";
    rpc.largeImageText = "Gran Turismo Sport";
    status = RPCStatus::Playing;
    Discord_UpdatePresence(&rpc);
}

void RPC::shutdown() {
    if (rpcEnabled) {
        rpcEnabled = false;
        Discord_ClearPresence();
        Discord_Shutdown();
    }
}

bool RPC::getRPCEnabled() {
    return rpcEnabled;
}

} // namespace DiscordRPCHandler
