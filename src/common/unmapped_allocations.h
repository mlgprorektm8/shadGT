#pragma once
#include <iterator>
#include <map>
#include <type_traits>

namespace Common {
// A physical allocation can back disjoint ranges. Every range must be retired.
template <typename Ranges, typename IsUnmapped>
auto FullyUnmappedAllocations(const Ranges& ranges, IsUnmapped&& is_unmapped) {
    using Handle = std::remove_cvref_t<decltype((*std::begin(ranges)).memory)>;
    std::map<Handle, bool> result;
    for (const auto& range : ranges) {
        auto [it, inserted] = result.try_emplace(range.memory, true);
        it->second &= is_unmapped(range);
    }
    return result;
}
} // namespace Common
