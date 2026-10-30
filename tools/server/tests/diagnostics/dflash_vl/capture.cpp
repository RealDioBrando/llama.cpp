#include "llama.h"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>

// This probe supports one sequence and five target feature taps.
static int injections = 0;
static bool captured = false;

static void dump(const std::string & path, const void * data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write((const char *) data, bytes);
}

extern "C" int32_t llama_decode(llama_context * ctx, llama_batch batch) {
    static auto original = (int32_t (*)(llama_context *, llama_batch)) dlsym(RTLD_NEXT, "llama_decode");
    const char * directory = std::getenv("DFLASH_CAPTURE_DIR");
    char architecture[32] = {};
    auto model = llama_get_model(ctx);
    llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture));
    const bool draft = std::string(architecture) == "dflash";
    if (!directory || !draft || captured || !std::filesystem::exists(std::string(directory) + "/ARMED")) {
        return original(ctx, batch);
    }
    const auto rope_type = llama_model_rope_type(model);
    if (rope_type != LLAMA_ROPE_TYPE_MROPE && rope_type != LLAMA_ROPE_TYPE_IMROPE) {
        std::abort();
    }
    std::filesystem::create_directories(directory);
    const std::string root(directory);
    const int32_t rows = batch.n_tokens;
    const auto before = llama_memory_seq_pos_max(llama_get_memory(ctx), 0);
    if (batch.embd) {
        const char * configured = std::getenv("DFLASH_CAPTURE_WIDTH");
        const auto width = configured ? std::atoi(configured) : 12800;
        if (width != 12800 || llama_model_n_embd(model) != 2560 || llama_model_n_layer(model) != 5) {
            std::abort();
        }
        const std::string prefix = root + "/inject_" + std::to_string(injections++);
        dump(prefix + ".bin", batch.embd, (size_t) rows * width * sizeof(float));
        dump(prefix + ".pos", batch.pos, (size_t) 4 * rows * sizeof(llama_pos));
        int spatial = 0;
        double square = 0;
        for (int i = 0; i < rows; ++i) {
            spatial += batch.pos[i] != batch.pos[rows + i] || batch.pos[i] != batch.pos[2 * rows + i];
        }
        for (size_t i = 0; i < (size_t) rows * width; ++i) {
            square += (double) batch.embd[i] * batch.embd[i];
        }
        const int32_t rc = original(ctx, batch);
        std::ofstream out(prefix + ".json");
        out << "{\"rows\":" << rows << ",\"width\":" << width
            << ",\"spatial_rows\":" << spatial << ",\"l2\":" << std::sqrt(square)
            << ",\"pos_before\":" << before << ",\"pos_after\":"
            << llama_memory_seq_pos_max(llama_get_memory(ctx), 0) << ",\"rc\":" << rc << "}";
        return rc;
    }
    if (batch.token) {
        dump(root + "/noise.tokens", batch.token, rows * sizeof(llama_token));
        dump(root + "/noise.pos", batch.pos, rows * sizeof(llama_pos));
        const int32_t rc = original(ctx, batch);
        if (rc == 0) {
            llama_synchronize(ctx);
            const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
            std::vector<float> logits;
            for (int i = 0; i < rows; ++i) {
                const auto values = llama_get_logits_ith(ctx, i);
                logits.insert(logits.end(), values, values + vocab);
            }
            dump(root + "/logits.bin", logits.data(), logits.size() * sizeof(float));
            std::ofstream out(root + "/noise.json");
            out << "{\"rows\":" << rows << ",\"vocab\":" << vocab
                << ",\"rope_type\":" << (int) llama_model_rope_type(model)
                << ",\"injection_batches\":" << injections << ",\"pos_before\":" << before << "}";
            captured = true;
        }
        return rc;
    }
    return original(ctx, batch);
}
