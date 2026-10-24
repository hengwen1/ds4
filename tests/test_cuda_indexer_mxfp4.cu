/* Blackwell block-scaled MXFP4 light-indexer scorer test.
 *
 * Drives ds4_gpu_indexer_scores_decode_batch_tensor with shapes that engage the
 * MXFP4 arm (>= 32 tokens, >= 1024 components, 64 heads x 128 dim, ratio 4) and
 * checks the result against a double-precision oracle that replicates the MXFP4
 * encode (E2M1 nearest-level, biased E8M0 scale) exactly. Both block-scaled
 * paths are covered: the sm_121a mma.sync scorer (DGX Spark / RTX 50) and the
 * sm_110a tcgen05 scorer (Jetson Thor). The other indexer arms are disabled by
 * environment so a dispatch miss fails the test instead of silently passing.
 *
 * Building without DS4_CUDA_HAVE_MXF4/DS4_CUDA_HAVE_TCGEN05 makes the MXFP4 arm
 * unavailable; the test then reports that instead of testing another kernel.
 *
 *   ./tests/test_cuda_indexer_mxfp4            correctness (20 shapes)
 *   ./tests/test_cuda_indexer_mxfp4 --bench    interleaved MXFP4 vs WMMA timing
 */
#include "ds4_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t state = 91234;
static float random_value() {
    state = state * 1664525u + 1013904223u;
    return ((int)(state >> 16) - 32768) / 4096.0f;
}

static float bits_to_float(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof(f));
    return f;
}

/* The encode kernel's nearest E2M1 level for |v| * 2^(254-exp). */
static float e2m1_value(unsigned code) {
    static const float table[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                    -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    return table[code & 15u];
}

static unsigned encode_exp(float amax) {
    if (amax <= 0.0f) return 127u;
    const float scaled = amax / 6.0f;
    uint32_t bits;
    memcpy(&bits, &scaled, sizeof(bits));
    uint32_t exponent = (bits >> 23u) & 0xffu;
    if ((bits & 0x7fffffu) != 0u) exponent++;
    return std::min(std::max(exponent, 1u), 254u);
}

static unsigned encode_level(float v, unsigned exp) {
    const float reciprocal = bits_to_float((254u - exp) << 23u);
    const float a = std::fabs(v) * reciprocal;
    if (a <= 0.25f) return 0u;
    if (a <= 0.75f) return 1u;
    if (a <= 1.25f) return 2u;
    if (a <= 1.75f) return 3u;
    if (a <= 2.5f) return 4u;
    if (a <= 3.5f) return 5u;
    if (a <= 5.0f) return 6u;
    return 7u;
}

static std::vector<double> ref_score;
static std::vector<double> ref_magnitude;

static void check(uint32_t rows, uint32_t keys, uint32_t start) {
    const uint32_t heads = 64, dim = 128, ratio = 4;
    const float scale = 0.011048543f, sentinel = -9182.0f;
    const size_t count = (size_t)rows * keys;
    std::vector<float> q((size_t)rows * heads * dim), w((size_t)rows * heads);
    std::vector<float> k((size_t)keys * dim), result(count + 16);
    for (float &v : q) v = random_value();
    for (float &v : w) v = random_value() * 0.25f;
    for (float &v : k) v = random_value();

    /* Double oracle over the same quantization the encode kernels apply. */
    ref_score.assign(count, 0.0);
    ref_magnitude.assign(count, 0.0);
    std::vector<unsigned> qexp((size_t)rows * heads * 4u), kexp((size_t)keys * 4u);
    std::vector<unsigned> qcode((size_t)rows * heads * dim), kcode((size_t)keys * dim);
    for (uint32_t r = 0; r < rows * heads; r++)
        for (uint32_t g = 0; g < 4u; g++) {
            float amax = 0.f;
            for (uint32_t e = 0; e < 32u; e++)
                amax = std::max(amax, std::fabs(q[(size_t)r * dim + g * 32u + e]));
            const unsigned exp = encode_exp(amax);
            qexp[(size_t)r * 4u + g] = exp;
            for (uint32_t e = 0; e < 32u; e++) {
                const float v = q[(size_t)r * dim + g * 32u + e];
                qcode[(size_t)r * dim + g * 32u + e] =
                    encode_level(v, exp) | (v < 0.f ? 8u : 0u);
            }
        }
    for (uint32_t r = 0; r < keys; r++)
        for (uint32_t g = 0; g < 4u; g++) {
            float amax = 0.f;
            for (uint32_t e = 0; e < 32u; e++)
                amax = std::max(amax, std::fabs(k[(size_t)r * dim + g * 32u + e]));
            const unsigned exp = encode_exp(amax);
            kexp[(size_t)r * 4u + g] = exp;
            for (uint32_t e = 0; e < 32u; e++) {
                const float v = k[(size_t)r * dim + g * 32u + e];
                kcode[(size_t)r * dim + g * 32u + e] =
                    encode_level(v, exp) | (v < 0.f ? 8u : 0u);
            }
        }
    for (uint32_t t = 0; t < rows; t++) {
        const uint32_t visible = std::min((start + t + 1) / ratio, keys);
        for (uint32_t j = 0; j < keys; j++) {
            double sum = 0, magnitude = 0;
            if (j < visible)
                for (uint32_t h = 0; h < heads; h++) {
                    double dot = 0;
                    for (uint32_t g = 0; g < 4u; g++) {
                        const float sq = bits_to_float(qexp[((size_t)(t * heads + h)) * 4u + g] << 23u);
                        const float sk = bits_to_float(kexp[(size_t)j * 4u + g] << 23u);
                        for (uint32_t e = 0; e < 32u; e++) {
                            const uint32_t d = g * 32u + e;
                            dot += (double)e2m1_value(qcode[((size_t)(t * heads + h)) * dim + d]) *
                                   (double)e2m1_value(kcode[(size_t)j * dim + d]) *
                                   (double)sq * (double)sk;
                        }
                    }
                    const double term = std::max(dot, 0.0) * w[(size_t)t * heads + h];
                    sum += term;
                    magnitude += std::fabs(term);
                }
            ref_score[(size_t)t * keys + j] = j >= visible ? -INFINITY : sum * (double)scale;
            ref_magnitude[(size_t)t * keys + j] = magnitude;
        }
    }

    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q.size() * sizeof(float));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(w.size() * sizeof(float));
    ds4_gpu_tensor *kt = ds4_gpu_tensor_alloc(k.size() * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(result.size() * sizeof(float));
    CHECK(qt && wt && kt && out);
    CHECK(ds4_gpu_tensor_write(qt, 0, q.data(), q.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(wt, 0, w.data(), w.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(kt, 0, k.data(), k.size() * sizeof(float)));
    CHECK(ds4_gpu_tensor_fill_f32(out, sentinel, result.size()));
    CHECK(ds4_gpu_indexer_scores_decode_batch_tensor(out, qt, wt, kt,
        keys, rows, start, heads, dim, ratio, scale));
    CHECK(ds4_gpu_tensor_read(out, 0, result.data(), result.size() * sizeof(float)));
    for (size_t i = count; i < result.size(); i++) CHECK(result[i] == sentinel);
    for (uint32_t t = 0; t < rows; t++) {
        const uint32_t visible = std::min((start + t + 1) / ratio, keys);
        for (uint32_t j = 0; j < keys; j++) {
            const float v = result[(size_t)t * keys + j];
            if (j >= visible) {
                CHECK(std::isinf(v) && v < 0);
                continue;
            }
            CHECK(!std::isnan(v));
            const double magnitude = ref_magnitude[(size_t)t * keys + j];
            const double error = std::fabs((double)v - ref_score[(size_t)t * keys + j]);
            if (!(error < 2e-5 * (1.0 + magnitude * (double)scale))) {
                printf("MISMATCH rows=%u keys=%u t=%u j=%u ref=%.6f got=%.6f mag=%.6f\n",
                       rows, keys, t, j, ref_score[(size_t)t * keys + j], (double)v, magnitude);
                CHECK(0);
            }
        }
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(kt);
    ds4_gpu_tensor_free(wt);
    ds4_gpu_tensor_free(qt);
}

struct bench_case { uint32_t tokens, comps; };

/* Interleaved A/B timing of the two indexer arms: DS4_CUDA_NO_INDEXER_MXF4=1
 * selects the established WMMA scorer, unset selects the MXFP4 arm. The two
 * run alternately so a clock ramp cannot favor either one; the best of each is
 * reported. pos0 puts the tokens at the end of the context, which is what
 * prefill and verification batches see (all components visible). */
static void bench(const bench_case *cases, size_t n_cases) {
    const uint32_t heads = 64, dim = 128, ratio = 4;
    const float scale = 0.011048543f;
    printf("%9s %9s %12s %12s %9s\n", "tokens", "comps", "wmma ms", "mxfp4 ms", "speedup");
    for (size_t ci = 0; ci < n_cases; ci++) {
        const uint32_t rows = cases[ci].tokens, keys = cases[ci].comps;
        const uint32_t start = keys * ratio > rows ? keys * ratio - rows : 0;
        std::vector<float> q((size_t)rows * heads * dim), w((size_t)rows * heads), k((size_t)keys * dim);
        state = 4711;
        for (float &v : q) v = random_value() * 0.25f;
        for (float &v : w) v = random_value() * 0.25f;
        for (float &v : k) v = random_value() * 0.25f;
        ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q.size() * sizeof(float));
        ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(w.size() * sizeof(float));
        ds4_gpu_tensor *kt = ds4_gpu_tensor_alloc(k.size() * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((size_t)rows * keys * sizeof(float));
        CHECK(qt && wt && kt && out);
        CHECK(ds4_gpu_tensor_write(qt, 0, q.data(), q.size() * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(wt, 0, w.data(), w.size() * sizeof(float)));
        CHECK(ds4_gpu_tensor_write(kt, 0, k.data(), k.size() * sizeof(float)));
        double best[2] = {1e9, 1e9};
        const int iters = (size_t)rows * keys > 8000000u ? 4 : 8;
        for (int it = 0; it < iters; it++) {
            for (int mode = 0; mode < 2; mode++) {
                if (mode == 0) setenv("DS4_CUDA_NO_INDEXER_MXF4", "1", 1);
                else unsetenv("DS4_CUDA_NO_INDEXER_MXF4");
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                cudaEventRecord(a);
                CHECK(ds4_gpu_indexer_scores_decode_batch_tensor(out, qt, wt, kt,
                    keys, rows, start, heads, dim, ratio, scale));
                cudaEventRecord(b);
                cudaEventSynchronize(b);
                float ms = 0;
                cudaEventElapsedTime(&ms, a, b);
                cudaEventDestroy(a);
                cudaEventDestroy(b);
                if (ms < best[mode]) best[mode] = ms;
            }
        }
        unsetenv("DS4_CUDA_NO_INDEXER_MXF4");
        printf("%9u %9u %12.3f %12.3f %8.2fx\n", rows, keys, best[0], best[1], best[0] / best[1]);
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(kt);
        ds4_gpu_tensor_free(wt);
        ds4_gpu_tensor_free(qt);
    }
}

int main(int argc, char **argv) {
    const bool do_bench = argc > 1 && strcmp(argv[1], "--bench") == 0;
    CHECK(ds4_gpu_init());
    ds4_gpu_set_quality(false);
    if (do_bench) {
        static const bench_case cases[] = {
            {32, 4096}, {128, 4096}, {256, 8192}, {512, 8192},
            {1024, 16384}, {2048, 16384},
        };
        bench(cases, sizeof(cases) / sizeof(cases[0]));
        ds4_gpu_cleanup();
        return 0;
    }
    /* Only the MXFP4 arm may answer: disable every fallback so a dispatch miss
     * fails the decode call instead of silently testing another kernel. */
    for (const char *name : {"DS4_CUDA_NO_INDEXER_WMMA",
                             "DS4_CUDA_NO_INDEXER_WMMA128", "DS4_CUDA_NO_INDEXER_WMMA64",
                             "DS4_CUDA_NO_INDEXER_DIRECT_ONE"})
        CHECK(setenv(name, "1", 1) == 0);
    unsigned cases = 0;
    for (uint32_t keys : {1024u, 1025u, 1536u, 4096u}) {
        for (uint32_t rows : {32u, 33u, 48u, 64u}) {
            check(rows, keys, keys * 4 > rows ? keys * 4 - rows : 0);
            cases++;
        }
        check(32u, keys, 0); /* Everything masked at the frontier. */
        cases++;
    }
    ds4_gpu_cleanup();
    printf("CUDA indexer MXFP4: %u cases, double quantized oracle PASS\n", cases);
    return 0;
}
