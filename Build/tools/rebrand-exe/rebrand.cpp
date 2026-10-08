// shadGT: replaces the icon and the version-info names of a Windows executable.
//   rebrand <exe> <icon.ico> <product name> <file description>
// Used on the prebuilt shadPS4 Qt launcher that ships as "shadGT Launcher.exe". Only the icon and
// the display strings change; the program itself is untouched.

#define UNICODE
#define _UNICODE
#include <windows.h>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#pragma pack(push, 2)
struct IconDirEntry {
    BYTE width, height, colors, reserved;
    WORD planes, bit_count;
    DWORD bytes;
    DWORD offset;
};
struct GroupIconEntry {
    BYTE width, height, colors, reserved;
    WORD planes, bit_count;
    DWORD bytes;
    WORD id;
};
#pragma pack(pop)

static BOOL CALLBACK CollectLang(HMODULE, LPCWSTR, LPCWSTR, WORD lang, LONG_PTR param) {
    reinterpret_cast<std::vector<WORD>*>(param)->push_back(lang);
    return TRUE;
}

static std::vector<WORD> Languages(const wchar_t* exe, LPCWSTR type, LPCWSTR name) {
    std::vector<WORD> langs;
    if (HMODULE m = LoadLibraryExW(exe, nullptr, LOAD_LIBRARY_AS_DATAFILE)) {
        EnumResourceLanguagesW(m, type, name, CollectLang, reinterpret_cast<LONG_PTR>(&langs));
        FreeLibrary(m);
    }
    return langs;
}

static BOOL CALLBACK FirstGroup(HMODULE, LPCWSTR, LPWSTR name, LONG_PTR param) {
    auto* out = reinterpret_cast<std::wstring*>(param);
    *out = IS_INTRESOURCE(name) ? L"#" + std::to_wstring(reinterpret_cast<ULONG_PTR>(name)) : name;
    return FALSE;
}

// VS_VERSIONINFO blocks: length, value length, type, key, padding, value, children.
static void Align(std::vector<BYTE>& b) {
    while (b.size() % 4) b.push_back(0);
}
static size_t Begin(std::vector<BYTE>& b, const std::wstring& key, WORD value_len, WORD type) {
    Align(b);
    const size_t start = b.size();
    WORD header[3] = {0, value_len, type};
    b.insert(b.end(), reinterpret_cast<BYTE*>(header), reinterpret_cast<BYTE*>(header) + 6);
    const auto* k = reinterpret_cast<const BYTE*>(key.c_str());
    b.insert(b.end(), k, k + (key.size() + 1) * 2);
    Align(b);
    return start;
}
static void End(std::vector<BYTE>& b, size_t start) {
    const WORD len = WORD(b.size() - start);
    memcpy(&b[start], &len, 2);
}
static void String(std::vector<BYTE>& b, const std::wstring& key, const std::wstring& value) {
    const size_t s = Begin(b, key, WORD(value.size() + 1), 1);
    const auto* v = reinterpret_cast<const BYTE*>(value.c_str());
    b.insert(b.end(), v, v + (value.size() + 1) * 2);
    End(b, s);
}

int wmain(int argc, wchar_t** argv) {
    if (argc < 5) {
        fwprintf(stderr, L"usage: rebrand <exe> <icon.ico> <product> <description>\n");
        return 1;
    }
    std::ifstream ico{argv[2], std::ios::binary};
    std::vector<BYTE> icon((std::istreambuf_iterator<char>(ico)), {});
    if (icon.size() < 6) {
        fwprintf(stderr, L"bad icon\n");
        return 1;
    }
    const WORD count = *reinterpret_cast<WORD*>(&icon[4]);
    const auto* entries = reinterpret_cast<IconDirEntry*>(&icon[6]);

    HMODULE module = LoadLibraryExW(argv[1], nullptr, LOAD_LIBRARY_AS_DATAFILE);
    std::wstring group = L"#1";
    if (module) {
        EnumResourceNamesW(module, RT_GROUP_ICON, FirstGroup, reinterpret_cast<LONG_PTR>(&group));
        FreeLibrary(module);
    }
    const LPCWSTR group_name = group[0] == L'#' ? MAKEINTRESOURCEW(_wtoi(group.c_str() + 1))
                                                : group.c_str();

    const auto group_langs = Languages(argv[1], RT_GROUP_ICON, group_name);
    const auto version_langs = Languages(argv[1], RT_VERSION, MAKEINTRESOURCEW(1));
    HANDLE update = BeginUpdateResourceW(argv[1], FALSE);
    if (!update) {
        fwprintf(stderr, L"BeginUpdateResource failed %lu\n", GetLastError());
        return 1;
    }
    const LANGID lang = MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL);
    std::vector<BYTE> group_data(6 + count * sizeof(GroupIconEntry));
    memcpy(group_data.data(), icon.data(), 6);
    constexpr WORD FirstIconId = 900;  // Above the ids Qt and the launcher use.
    for (WORD i = 0; i < count; ++i) {
        const auto& e = entries[i];
        UpdateResourceW(update, RT_ICON, MAKEINTRESOURCEW(FirstIconId + i), lang,
                        &icon[e.offset], e.bytes);
        GroupIconEntry g{e.width, e.height, e.colors, e.reserved, e.planes, e.bit_count, e.bytes,
                         WORD(FirstIconId + i)};
        memcpy(&group_data[6 + i * sizeof(g)], &g, sizeof(g));
    }
    // Replace the group in every language it was stored under.
    for (const WORD l : group_langs) {
        UpdateResourceW(update, RT_GROUP_ICON, group_name, l, nullptr, 0);
    }
    UpdateResourceW(update, RT_GROUP_ICON, group_name, MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US),
                    group_data.data(), DWORD(group_data.size()));

    // Version info with the new names.
    std::vector<BYTE> v;
    VS_FIXEDFILEINFO fixed{};
    fixed.dwSignature = 0xFEEF04BD;
    fixed.dwStrucVersion = 0x10000;
    fixed.dwFileVersionMS = fixed.dwProductVersionMS = 0x10000;
    fixed.dwFileOS = VOS_NT_WINDOWS32;
    fixed.dwFileType = VFT_APP;
    const size_t root = Begin(v, L"VS_VERSION_INFO", sizeof(fixed), 0);
    v.insert(v.end(), reinterpret_cast<BYTE*>(&fixed), reinterpret_cast<BYTE*>(&fixed) + sizeof(fixed));
    const size_t sfi = Begin(v, L"StringFileInfo", 0, 1);
    const size_t table = Begin(v, L"040904b0", 0, 1);
    String(v, L"CompanyName", L"shadGT");
    String(v, L"FileDescription", argv[4]);
    String(v, L"FileVersion", L"1.0");
    String(v, L"InternalName", L"shadGT Launcher");
    String(v, L"OriginalFilename", L"shadGT Launcher.exe");
    String(v, L"ProductName", argv[3]);
    String(v, L"ProductVersion", L"1.0");
    End(v, table);
    End(v, sfi);
    const size_t vfi = Begin(v, L"VarFileInfo", 0, 1);
    const size_t trans = Begin(v, L"Translation", 4, 0);
    const WORD translation[2] = {0x0409, 0x04b0};
    v.insert(v.end(), reinterpret_cast<const BYTE*>(translation),
             reinterpret_cast<const BYTE*>(translation) + 4);
    End(v, trans);
    End(v, vfi);
    End(v, root);
    for (const WORD l : version_langs) {
        UpdateResourceW(update, RT_VERSION, MAKEINTRESOURCEW(1), l, nullptr, 0);
    }
    UpdateResourceW(update, RT_VERSION, MAKEINTRESOURCEW(1),
                    MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), v.data(), DWORD(v.size()));
    if (!EndUpdateResourceW(update, FALSE)) {
        fwprintf(stderr, L"EndUpdateResource failed %lu\n", GetLastError());
        return 1;
    }
    wprintf(L"rebranded %ls (icon group %ls, %u images)\n", argv[1], group.c_str(), count);
    return 0;
}
