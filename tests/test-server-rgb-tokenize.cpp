// Tests the failure contract of server_context::tokenize_with_rgb_image.
//
// The success path (tokenizing a real image) needs a mmproj encoder, which no
// test fixture provides, so here we verify the documented guarantees instead:
// without a loaded projector, or with invalid arguments, the call returns an
// empty server_tokens and does not throw.

#include "server-context.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define EXPECT_TRUE(x)                                                       \
    do {                                                                     \
        if (!(x)) {                                                          \
            fprintf(stderr, "\n  FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

static void test_no_mmproj_returns_empty() {
    server_context ctx;

    // 4x4 solid white, tightly packed RGB
    std::vector<unsigned char> pixels(4 * 4 * 3, 0xff);

    auto tokens = ctx.tokenize_with_rgb_image("describe this image", pixels.data(), 4, 4);
    EXPECT_TRUE(tokens.size() == 0);
    EXPECT_TRUE(!tokens.has_mtmd);
}

static void test_bad_args_return_empty() {
    server_context ctx;

    std::vector<unsigned char> pixels(2 * 2 * 3, 0x00);

    EXPECT_TRUE(ctx.tokenize_with_rgb_image("hi", nullptr, 2, 2).size() == 0);
    EXPECT_TRUE(ctx.tokenize_with_rgb_image("hi", pixels.data(), 0, 2).size() == 0);
    EXPECT_TRUE(ctx.tokenize_with_rgb_image("hi", pixels.data(), 2, 0).size() == 0);
}

int main() {
    fprintf(stderr, "test-server-rgb-tokenize: guard path\n");
    test_no_mmproj_returns_empty();
    test_bad_args_return_empty();
    fprintf(stderr, "test-server-rgb-tokenize: ALL PASSED\n");
    return 0;
}
