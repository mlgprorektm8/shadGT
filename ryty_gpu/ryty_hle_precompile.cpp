// Non-renderer host libraries share system-service stubs but do not own shader
// compilation. Only libSceVideoOut implements and holds the game's precompile.
namespace Vulkan {
bool PrecompileHoldsGame() {
    return false;
}
}
