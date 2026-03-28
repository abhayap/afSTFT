# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

afSTFT is an Alias-free Short-Time Fourier Transform library by Juha Vilkamo. It provides a complex-modulated filter bank for time-frequency audio processing with built-in anti-aliasing, similar to MPEG filter banks but with standard STFT band center frequencies. Licensed under MIT.

This is a fork (origin: abhayap/afSTFT) of jvilkamo/afSTFT (upstream).

## Build

Compile the MEX file from MATLAB:

```matlab
mex src/afSTFT_mex.c src/afSTFTlib.c src/vecTools.c src/fft4g.c -Isrc COPTIMFLAGS='-O3 -DNDEBUG' -output afSTFT
```

Or run `compile_afSTFT_mex.m` which also includes a performance benchmark.

On macOS, add `-DVDSP` to COPTIMFLAGS and link `-framework Accelerate` to use Apple's vDSP FFT (~2x faster than Ooura). The precompiled `afSTFT.mexmaci64` already uses vDSP.

## Architecture

### Core C Library (src/)

- **afSTFTlib.h/c** — Main library. Defines the `afSTFT` and `afHybrid` structs and the four API functions: `afSTFTinit()`, `afSTFTforward()`, `afSTFTinverse()`, `afSTFTfree()`. Processing uses 10 overlapping hops with a pre-computed prototype filter from `afSTFT_protoFilter.h`.
- **afSTFT_mex.c** — MATLAB MEX gateway. Multiplexes init/forward/inverse/free through a single `afSTFT()` MEX function based on input argument shape. Persistent state via `mexMakeMemoryPersistent`.
- **vecTools.h/c** — Vector math utilities (`vtClr`, `vtVma`) and FFT wrappers. Uses `#ifdef VDSP` to switch between Apple Accelerate and Ooura backends. Contains a phase conjugation fix for Ooura FFT compatibility.
- **fft4g.h/c** — Ooura's pure-C FFT implementation (fallback when vDSP unavailable).
- **afSTFT_protoFilter.h** — Pre-computed prototype filter coefficients for both standard and low-delay modes (large data table, ~2900 lines).

### Processing Modes

- **Standard**: 129 frequency bands (DC through Nyquist)
- **Hybrid**: 133 bands — subdivides first 4 bands using half-band filters for better low-frequency resolution. Only for hop sizes 64 and 128.
- **Low-delay**: Phase adjustment for reduced latency

Supported hop sizes: 32, 64, 128, 256, 512, 1024 samples.

### MATLAB Interface

- **afSTFT_usage_instruction.m** — Reference examples for single-shot and continuous processing
- **afAnalyze.m / afSynthesize.m** — Convenience wrappers for single-shot analysis/synthesis

The MEX function behavior is determined by input:
- `afSTFT(hopSize, fwdCh, invCh, ...)` — initialize
- `afSTFT(timeSignal)` where input is (samples × channels) — forward transform
- `afSTFT(freqData)` where input is (bands × frames × channels) — inverse transform
- `afSTFT()` — free resources
