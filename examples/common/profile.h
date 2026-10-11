#ifndef __EXAMPLES_COMMON_PROFILE_H__
#define __EXAMPLES_COMMON_PROFILE_H__

#include <cstdint>
#include <string>

#include "common.h"

// Resource profiles: a few named sets of options that trade speed for a smaller memory footprint.
// A profile only sets options the user did not give on the command line.
namespace sd_profile {

    // From the lightest on resources to the heaviest. The levels after LITE are not selectable from
    // the command line: a server steps down to them when a request runs out of memory.
    enum class Level {
        MAX,        // the defaults of the engine
        BALANCED,   // memory guard at 90
        LITE,       // + text encoder on the CPU, tiled VAE, unbatched CFG, memory guard at 85
        LITE_DISK,  // + weights streamed from disk
        LITE_CPU,   // + every module computed on the CPU
    };

    const char* level_name(Level level);

    // Accepts max, balanced, lite and auto; for auto, automatic is set and level is left unchanged.
    bool parse(const std::string& text, Level& level, bool& automatic);

    // The next level that needs less memory; the last level returns itself.
    Level lighter(Level level);

    struct Resources {
        int64_t model_bytes   = 0;   // size of the model files, 0 when unknown
        int64_t ram_available = -1;  // RAM a program can still use, -1 when unknown
        int64_t gpu_free      = 0;   // free memory of the discrete GPU with the most, 0 without one
    };

    // Reads the sizes of the model files and the memory of the machine. Reads no model data.
    Resources probe(const SDContextParams& ctx);

    // Picks MAX, BALANCED or LITE: MAX when the weights and the working memory fit with room to
    // spare in the memory where the compute happens, BALANCED when they fit, LITE otherwise.
    Level choose(const Resources& resources, std::string* reason = nullptr);

    // Sets the options of the level. With respect_explicit, an option found on the command line is
    // left as it is; without it (a server recovering from a failure) the level takes precedence.
    void apply(Level level, SDContextParams& ctx, SDGenerationParams& gen, bool respect_explicit);

    // Resolves --profile (when present) and applies it. Returns false after logging an error when the
    // name is not valid. applied receives the level in effect (MAX without --profile).
    bool apply_from_options(SDContextParams& ctx, SDGenerationParams& gen, Level& applied);

}  // namespace sd_profile

#endif  // __EXAMPLES_COMMON_PROFILE_H__
