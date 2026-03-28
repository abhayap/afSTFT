/*
 * Comprehensive test suite for the afSTFT library.
 *
 * Build (portable):
 *   cc -o test_afstft test_afstft.c src/afSTFTlib.c src/vecTools.c src/fft4g.c -Isrc -lm -O2
 *
 * Build (macOS with vDSP):
 *   cc -o test_afstft test_afstft.c src/afSTFTlib.c src/vecTools.c -Isrc -lm -O2 -DVDSP -framework Accelerate
 *
 * Run:
 *   ./test_afstft
 *
 * Exits 0 on all-pass, 1 on any failure.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "afSTFTlib.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define CHECK(cond, name) do { \
    if (cond) { printf("  PASS: %s\n", name); tests_passed++; } \
    else      { printf("  FAIL: %s\n", name); tests_failed++; } \
} while (0)

/* ---- Simple LCG PRNG for reproducibility ---- */
static uint32_t rng_state = 12345;

static float randf(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (float)(rng_state >> 16) / 32768.0f - 1.0f;
}

static void seed_rng(uint32_t s) { rng_state = s; }

/* ---- Allocation helpers ---- */

static complexVector *alloc_fd(int nCh, int nBands) {
    complexVector *fd = (complexVector *)malloc(sizeof(complexVector) * nCh);
    for (int ch = 0; ch < nCh; ch++) {
        fd[ch].re = (float *)calloc(nBands, sizeof(float));
        fd[ch].im = (float *)calloc(nBands, sizeof(float));
    }
    return fd;
}

static void free_fd(complexVector *fd, int nCh) {
    for (int ch = 0; ch < nCh; ch++) {
        free(fd[ch].re);
        free(fd[ch].im);
    }
    free(fd);
}

static float **alloc_td(int nCh, int len) {
    float **td = (float **)malloc(sizeof(float *) * nCh);
    for (int ch = 0; ch < nCh; ch++)
        td[ch] = (float *)calloc(len, sizeof(float));
    return td;
}

static void free_td(float **td, int nCh) {
    for (int ch = 0; ch < nCh; ch++)
        free(td[ch]);
    free(td);
}

/* ==================================================================
 *  1. Unit tests
 * ================================================================== */

static void test_vtClr(void) {
    printf("\n--- vtClr ---\n");
    float buf[16];
    for (int i = 0; i < 16; i++) buf[i] = (float)(i + 1);
    vtClr(buf, 16);
    int ok = 1;
    for (int i = 0; i < 16; i++)
        if (buf[i] != 0.0f) { ok = 0; break; }
    CHECK(ok, "zeroes all elements");
}

static void test_vtVma(void) {
    printf("\n--- vtVma ---\n");
    float a[4] = {1, 2, 3, 4};
    float b[4] = {5, 6, 7, 8};
    float c[4] = {10, 20, 30, 40};
    vtVma(a, b, c, 4);
    /* c[i] += a[i]*b[i]: 15, 32, 51, 72 */
    CHECK(c[0] == 15.0f && c[1] == 32.0f && c[2] == 51.0f && c[3] == 72.0f,
          "multiply-accumulate with known values");
}

static void test_vtNegStride2(void) {
    printf("\n--- vtNegStride2 ---\n");
    float v[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    vtNegStride2(v, 4); /* negate indices 0, 2, 4, 6 */
    CHECK(v[0] == -1.0f && v[1] == 2.0f &&
          v[2] == -3.0f && v[3] == 4.0f &&
          v[4] == -5.0f && v[5] == 6.0f &&
          v[6] == -7.0f && v[7] == 8.0f,
          "negates every other element");
}

static void test_fft_roundtrip(void) {
    printf("\n--- FFT round-trip ---\n");
    int hopSizes[] = {32, 64, 128, 256, 512, 1024};

    for (int h = 0; h < 6; h++) {
        int hopSize = hopSizes[h];
        int N = hopSize * 2;
        int log2n = 0;
        while ((1 << log2n) < N) log2n++;

        float *td = (float *)calloc(N, sizeof(float));
        float *fd = (float *)calloc(N + 2, sizeof(float));
        void *plan = NULL;
        vtInitFFT(&plan, td, fd, log2n);

        /* Fill with pseudo-random signal */
        seed_rng(42);
        float *orig = (float *)malloc(sizeof(float) * N);
        for (int i = 0; i < N; i++) {
            td[i] = randf();
            orig[i] = td[i];
        }

        vtRunFFT(plan, 1);  /* forward */
        vtRunFFT(plan, -1); /* inverse */

        /* Measure scale factor and reconstruction error */
        float scale = (fabsf(orig[0]) > 1e-10f) ? td[0] / orig[0] : 0.0f;
        float maxErr = 0;
        for (int i = 0; i < N; i++) {
            float err = fabsf(td[i] - orig[i] * scale);
            if (err > maxErr) maxErr = err;
        }
        float relErr = maxErr / (fabsf(scale) + 1e-30f);

        char name[80];
        snprintf(name, sizeof(name),
                 "hopSize=%d (scale=%.1f, maxRelErr=%.2e)", hopSize, scale, relErr);
        CHECK(relErr < 1e-4f, name);

        free(orig);
        vtFreeFFT(plan);
        free(td);
        free(fd);
    }
}

static void test_invalid_hopsize(void) {
    printf("\n--- Invalid hopSize ---\n");
    void *handle = NULL;

    afSTFTinit(&handle, 100, 1, 1, 0, 0);
    CHECK(handle == NULL, "hopSize=100 returns NULL");

    afSTFTinit(&handle, 0, 1, 1, 0, 0);
    CHECK(handle == NULL, "hopSize=0 returns NULL");
}

/* ==================================================================
 *  2. Property tests
 * ================================================================== */

static void test_dc_nyquist_imag(void) {
    printf("\n--- DC/Nyquist imaginary = 0 ---\n");
    int hopSize = 128;
    int nBands = hopSize + 1;
    void *handle = NULL;
    afSTFTinit(&handle, hopSize, 1, 1, 0, 0);

    float **inTD = alloc_td(1, hopSize);
    complexVector *outFD = alloc_fd(1, nBands);

    seed_rng(99);
    for (int hop = 0; hop < 20; hop++) {
        for (int i = 0; i < hopSize; i++) inTD[0][i] = randf();
        afSTFTforward(handle, inTD, outFD);
    }
    CHECK(outFD[0].im[0] == 0.0f, "DC imaginary == 0");
    CHECK(outFD[0].im[hopSize] == 0.0f, "Nyquist imaginary == 0");

    free_fd(outFD, 1);
    free_td(inTD, 1);
    afSTFTfree(handle);
}

static void test_channel_independence(void) {
    printf("\n--- Channel independence ---\n");
    int hopSize = 128;
    int nBands = hopSize + 1;
    void *handle = NULL;
    afSTFTinit(&handle, hopSize, 2, 2, 0, 0);

    float **inTD = alloc_td(2, hopSize);
    complexVector *outFD = alloc_fd(2, nBands);

    seed_rng(77);
    float maxCh1 = 0;
    for (int hop = 0; hop < 20; hop++) {
        for (int i = 0; i < hopSize; i++) {
            inTD[0][i] = randf();
            inTD[1][i] = 0.0f;
        }
        afSTFTforward(handle, inTD, outFD);
        for (int b = 0; b < nBands; b++) {
            float mag = fabsf(outFD[1].re[b]) + fabsf(outFD[1].im[b]);
            if (mag > maxCh1) maxCh1 = mag;
        }
    }
    char name[80];
    snprintf(name, sizeof(name),
             "ch1 stays silent (max=%.2e)", maxCh1);
    CHECK(maxCh1 < 1e-6f, name);

    free_fd(outFD, 2);
    free_td(inTD, 2);
    afSTFTfree(handle);
}

static void test_linearity(void) {
    printf("\n--- Linearity ---\n");
    int hopSize = 128;
    int nBands = hopSize + 1;
    float scale = 3.5f;
    int nHops = 15;

    void *h1 = NULL, *h2 = NULL;
    afSTFTinit(&h1, hopSize, 1, 1, 0, 0);
    afSTFTinit(&h2, hopSize, 1, 1, 0, 0);

    float **in1 = alloc_td(1, hopSize);
    float **in2 = alloc_td(1, hopSize);
    complexVector *out1 = alloc_fd(1, nBands);
    complexVector *out2 = alloc_fd(1, nBands);

    float maxErr = 0;
    seed_rng(55);
    for (int hop = 0; hop < nHops; hop++) {
        for (int i = 0; i < hopSize; i++) {
            float v = randf();
            in1[0][i] = v;
            in2[0][i] = v * scale;
        }
        afSTFTforward(h1, in1, out1);
        afSTFTforward(h2, in2, out2);

        for (int b = 0; b < nBands; b++) {
            float errRe = fabsf(out2[0].re[b] - out1[0].re[b] * scale);
            float errIm = fabsf(out2[0].im[b] - out1[0].im[b] * scale);
            if (errRe > maxErr) maxErr = errRe;
            if (errIm > maxErr) maxErr = errIm;
        }
    }
    char name[80];
    snprintf(name, sizeof(name),
             "forward(%.1f*x) == %.1f*forward(x), maxErr=%.2e", scale, scale, maxErr);
    CHECK(maxErr < 1e-4f, name);

    free_fd(out1, 1); free_fd(out2, 1);
    free_td(in1, 1);  free_td(in2, 1);
    afSTFTfree(h1);   afSTFTfree(h2);
}

/* ==================================================================
 *  3. End-to-end round-trip tests
 * ================================================================== */

static float compute_snr(const float *ref, const float *test, int len) {
    double sig = 0, noise = 0;
    for (int i = 0; i < len; i++) {
        sig   += (double)ref[i]  * ref[i];
        noise += (double)(test[i] - ref[i]) * (test[i] - ref[i]);
    }
    if (noise < 1e-30) return 999.0f;
    return (float)(10.0 * log10(sig / noise));
}

static int find_delay(const float *ref, const float *test,
                      int refLen, int testLen, int maxDelay) {
    double bestCorr = -1e30;
    int bestD = 0;
    for (int d = 0; d < maxDelay && d < testLen; d++) {
        double corr = 0;
        int len = refLen;
        if (d + len > testLen) len = testLen - d;
        for (int i = 0; i < len; i++)
            corr += (double)ref[i] * test[i + d];
        if (corr > bestCorr) {
            bestCorr = corr;
            bestD = d;
        }
    }
    return bestD;
}

static void run_roundtrip(int hopSize, int LDmode, int hybridMode, float minSNR) {
    int nCh = 1;
    int nBands = hopSize + 1 + (hybridMode ? 4 : 0);
    int nHops = 80;
    int sigLen = nHops * hopSize;

    void *handle = NULL;
    afSTFTinit(&handle, hopSize, nCh, nCh, LDmode, hybridMode);
    if (!handle) {
        char msg[128];
        snprintf(msg, sizeof(msg),
                 "init returned NULL for hopSize=%d LD=%d hybrid=%d",
                 hopSize, LDmode, hybridMode);
        printf("  FAIL: %s\n", msg);
        tests_failed++;
        return;
    }

    float *input  = (float *)malloc(sizeof(float) * sigLen);
    float *output = (float *)calloc(sigLen, sizeof(float));
    seed_rng((uint32_t)(hopSize * 1000 + LDmode * 100 + hybridMode));
    for (int i = 0; i < sigLen; i++) input[i] = randf();

    float **inTD  = alloc_td(nCh, hopSize);
    float **outTD = alloc_td(nCh, hopSize);
    complexVector *fd = alloc_fd(nCh, nBands);

    for (int hop = 0; hop < nHops; hop++) {
        memcpy(inTD[0], &input[hop * hopSize], sizeof(float) * hopSize);
        afSTFTforward(handle, inTD, fd);
        afSTFTinverse(handle, fd, outTD);
        memcpy(&output[hop * hopSize], outTD[0], sizeof(float) * hopSize);
    }

    /* Find system delay via cross-correlation */
    int maxDelay = 20 * hopSize;
    if (maxDelay > sigLen / 2) maxDelay = sigLen / 2;
    int delay = find_delay(input, output, sigLen / 2, sigLen, maxDelay);

    /* Compute SNR over valid region (skip tail transient) */
    int validLen = sigLen - delay - hopSize * 5;
    if (validLen < hopSize * 10) validLen = hopSize * 10;
    if (delay + validLen > sigLen) validLen = sigLen - delay;

    float snr = compute_snr(input, &output[delay], validLen);

    char name[128];
    snprintf(name, sizeof(name),
             "hopSize=%d LD=%d hybrid=%d: delay=%d SNR=%.1f dB (min %.0f)",
             hopSize, LDmode, hybridMode, delay, snr, minSNR);
    CHECK(snr > minSNR, name);

    free(input);
    free(output);
    free_fd(fd, nCh);
    free_td(inTD, nCh);
    free_td(outTD, nCh);
    afSTFTfree(handle);
}

static void test_roundtrips(void) {
    printf("\n--- Round-trip tests ---\n");

    /* Standard mode — all 6 hop sizes */
    int hopSizes[] = {32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < 6; i++)
        run_roundtrip(hopSizes[i], 0, 0, 40.0f);

    /* Low-delay mode */
    run_roundtrip(128, 1, 0, 40.0f);

    /* Hybrid mode (valid for hopSize 64 and 128) */
    run_roundtrip(64,  0, 1, 40.0f);
    run_roundtrip(128, 0, 1, 40.0f);

    /* Combined low-delay + hybrid */
    run_roundtrip(64,  1, 1, 40.0f);
    run_roundtrip(128, 1, 1, 40.0f);
}

/* ==================================================================
 *  Main
 * ================================================================== */

int main(void) {
    printf("afSTFT Test Suite\n");
    printf("=================\n");

    /* Unit tests */
    test_vtClr();
    test_vtVma();
    test_vtNegStride2();
    test_fft_roundtrip();
    test_invalid_hopsize();

    /* Property tests */
    test_dc_nyquist_imag();
    test_channel_independence();
    test_linearity();

    /* Round-trip tests */
    test_roundtrips();

    /* Summary */
    printf("\n=================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
