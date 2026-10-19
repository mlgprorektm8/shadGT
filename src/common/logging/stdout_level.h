#pragma once
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace Common::Log::StdoutLevel {
inline constexpr unsigned Warning = 3;
inline constexpr std::array<std::string_view, 6> Names{"Trace", "Debug", "Info", "Warning", "Error", "Critical"};

inline std::string_view Trim(std::string_view text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    return first == text.npos ? std::string_view{} : text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

inline unsigned Parse(std::string_view text) {
    text = Trim(text);
    for (unsigned level = 0; level < Names.size(); ++level) {
        if (text.size() != Names[level].size()) continue;
        bool equal = true;
        for (unsigned i = 0; i < text.size(); ++i)
            equal &= std::tolower(static_cast<unsigned char>(text[i])) ==
                     std::tolower(static_cast<unsigned char>(Names[level][i]));
        if (equal) return level;
    }
    return Warning;
}

inline unsigned Read(const std::filesystem::path& ini) {
    std::ifstream input(ini);
    std::string line;
    while (std::getline(input, line)) {
        const auto separator = line.find('=');
        if (separator != std::string::npos && Trim(std::string_view(line).substr(0, separator)) == "stdout_level")
            return Parse(std::string_view(line).substr(separator + 1));
    }
    return Warning;
}

inline bool Enabled(unsigned message, unsigned minimum) { return message >= minimum; }
}
