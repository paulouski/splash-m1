#include "metal/DeviceCapabilities.hpp"

#include <string>

namespace splash {

std::string DeviceCapabilities::macosVersion() const {
    return std::to_string(macosMajor) + '.' + std::to_string(macosMinor) + '.' +
           std::to_string(macosPatch);
}

std::optional<std::string> DeviceCapabilities::validationError() const {
    // The operating system explains a missing placement-sparse query, so it
    // is reported ahead of every device feature.
#if defined(SPLASH_MACOS15_BUILD)
    if (!meetsMinimumMacos()) return "macos_15_required";
#else
    if (!meetsMinimumMacos()) return "macos_26_4_required";
#endif
    if (!physicalMemoryBytes) return "physical_memory_unavailable";
    if (!recommendedMaxWorkingSetBytes) {
        return "recommended_working_set_unavailable";
    }
    if (recommendedMaxWorkingSetBytes > physicalMemoryBytes) {
        return "recommended_working_set_exceeds_physical_memory";
    }
    if (!maxBufferLengthBytes) return "max_buffer_length_unavailable";
    if (appleGpuFamily < kMinimumAppleGpuFamily) return "apple_gpu_family_7_required";
    if (maxThreadgroupMemoryBytes < 32 * 1024) {
        return "threadgroup_memory_below_32_kib";
    }
    if (maxThreadgroupWidth < 256) {
        return "threadgroup_width_below_256";
    }
    if (!hasUnifiedMemory) return "unified_memory_required";
    // kRequiresPlacementSparse is false only on the macOS-15 build target,
    // where supportsPlacementSparse is always false (unqueryable below macOS
    // 26.4): the memory-plan worker must fall back to dense KV instead of
    // relying on this validation to reject the device.
    if (kRequiresPlacementSparse && !supportsPlacementSparse) {
        return "placement_sparse_required";
    }
    return std::nullopt;
}

std::optional<std::string> DeviceCapabilities::validationMessage() const {
    const std::optional<std::string> error = validationError();
    if (!error) return std::nullopt;
    // People know their chip, not its GPU family.
    static_assert(kMinimumAppleGpuFamily == 7, "name the family's first chip");
    const std::string family =
        appleGpuFamily ? "Apple GPU family " + std::to_string(appleGpuFamily)
                       : "no known Apple GPU family";
    const char *sparse = !meetsMinimumMacos()
                             ? "where placement-sparse support cannot be queried"
                         : supportsPlacementSparse
                             ? "with placement-sparse buffers"
                             : "without placement-sparse buffers";
#if defined(SPLASH_MACOS15_BUILD)
    const char *sparseRequirement = "";
#else
    const char *sparseRequirement = ", with placement-sparse buffers";
#endif
    return "Splash needs Apple GPU family " +
           std::to_string(kMinimumAppleGpuFamily) +
           " or newer (M1 or later) on macOS " +
           std::to_string(kMinimumMacosMajor) + '.' +
           std::to_string(kMinimumMacosMinor) +
           " or newer" + sparseRequirement + "; this Mac has " +
           deviceName + " (" + family + ") on macOS " + macosVersion() + ", " +
           sparse + " (" + *error + ')';
}

} // namespace splash
