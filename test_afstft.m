function test_afstft
% Comprehensive test suite for the afSTFT MATLAB/Octave interface.
%
% Requires the compiled MEX file (run compile_afSTFT_mex.m first).
%
% Usage:
%   test_afstft          % runs all tests, prints results, errors on failure
%
% Tests cover:
%   1. Single-shot round-trip via afAnalyze / afSynthesize
%   2. Continuous (loop) processing round-trip
%   3. Multiple configurations (hop sizes, hybrid, low-delay)
%   4. Multi-channel processing
%   5. Structural properties (DC/Nyquist, linearity, band count)

    passed = 0;
    failed = 0;

    function pass(name)
        fprintf('  PASS: %s\n', name);
        passed = passed + 1;
    end

    function fail(name)
        fprintf('  FAIL: %s\n', name);
        failed = failed + 1;
    end

    function check(cond, name)
        if cond, pass(name); else, fail(name); end
    end

    fprintf('afSTFT MATLAB Test Suite\n');
    fprintf('========================\n');

    %% ---- 1. Single-shot round-trip (afAnalyze / afSynthesize) ----
    fprintf('\n--- Single-shot round-trip (hybrid) ---\n');

    rng(42);
    N = 40000;
    nCh = 2;
    inputSig = randn(N, nCh);

    fd = afAnalyze(inputSig);
    outputSig = afSynthesize(fd);

    % Find delay and compute SNR per channel
    for ch = 1:nCh
        [snr, delay] = measure_snr(inputSig(:,ch), outputSig(:,ch));
        check(snr > 40, sprintf('ch%d: delay=%d SNR=%.1f dB', ch, delay, snr));
    end

    fprintf('\n--- Single-shot round-trip (non-hybrid) ---\n');
    fd_nh = afAnalyze(inputSig, 0);
    outputSig_nh = afSynthesize(fd_nh);
    for ch = 1:nCh
        [snr, delay] = measure_snr(inputSig(:,ch), outputSig_nh(:,ch));
        check(snr > 40, sprintf('ch%d: delay=%d SNR=%.1f dB', ch, delay, snr));
    end

    %% ---- 2. Continuous processing round-trip ----
    fprintf('\n--- Continuous processing round-trip ---\n');

    configs = {
        128, 'standard',   {};
        128, 'hybrid',     {'hybrid'};
        128, 'low_delay',  {'low_delay'};
        128, 'hybrid+LD',  {'hybrid', 'low_delay'};
        64,  'hop64',      {};
        64,  'hop64_hyb',  {'hybrid'};
        256, 'hop256',     {};
    };

    rng(123);
    for ci = 1:size(configs, 1)
        hopSize  = configs{ci, 1};
        tag      = configs{ci, 2};
        opts     = configs{ci, 3};
        sigLen   = hopSize * 80;
        inSig    = randn(sigLen, 1);
        outSig   = run_continuous(inSig, hopSize, 1, 1, opts);
        [snr, delay] = measure_snr(inSig, outSig);
        check(snr > 40, sprintf('hop=%d %s: delay=%d SNR=%.1f dB', ...
            hopSize, tag, delay, snr));
    end

    %% ---- 3. Multi-channel continuous ----
    fprintf('\n--- Multi-channel continuous ---\n');

    hopSize = 128;
    nCh = 4;
    sigLen = hopSize * 60;
    rng(77);
    inSig = randn(sigLen, nCh);
    outSig = run_continuous(inSig, hopSize, nCh, nCh, {});

    for ch = 1:nCh
        [snr, delay] = measure_snr(inSig(:,ch), outSig(:,ch));
        check(snr > 40, sprintf('ch%d: delay=%d SNR=%.1f dB', ch, delay, snr));
    end

    %% ---- 4. Band count verification ----
    fprintf('\n--- Band count ---\n');

    freqs = afSTFT(128, 1, 1);
    check(length(freqs) == 129, sprintf('standard 128: %d bands (expect 129)', length(freqs)));
    afSTFT();

    freqs = afSTFT(128, 1, 1, 'hybrid');
    check(length(freqs) == 133, sprintf('hybrid 128: %d bands (expect 133)', length(freqs)));
    afSTFT();

    freqs = afSTFT(64, 1, 1);
    check(length(freqs) == 65, sprintf('standard 64: %d bands (expect 65)', length(freqs)));
    afSTFT();

    freqs = afSTFT(64, 1, 1, 'hybrid');
    check(length(freqs) == 69, sprintf('hybrid 64: %d bands (expect 69)', length(freqs)));
    afSTFT();

    %% ---- 5. DC and Nyquist imaginary = 0 ----
    fprintf('\n--- DC/Nyquist imaginary ---\n');

    hopSize = 128;
    nBands = hopSize + 1;
    afSTFT(hopSize, 1, 1);
    rng(99);
    inFrame = randn(hopSize * 10, 1);
    fd = afSTFT(inFrame);   % (bands x frames x channels)
    dc_im  = max(abs(imag(fd(1, :))));
    nyq_im = max(abs(imag(fd(nBands, :))));
    afSTFT();
    check(dc_im == 0,  sprintf('DC imag max = %.2e', dc_im));
    check(nyq_im == 0, sprintf('Nyquist imag max = %.2e', nyq_im));

    %% ---- 6. Linearity ----
    fprintf('\n--- Linearity ---\n');

    hopSize = 128;
    alpha = 3.5;
    afSTFT(hopSize, 1, 1);
    rng(55);
    inFrame = randn(hopSize * 10, 1);
    fd1 = afSTFT(inFrame);
    afSTFT();

    afSTFT(hopSize, 1, 1);
    fd2 = afSTFT(alpha * inFrame);
    afSTFT();

    maxErr = max(abs(fd2(:) - alpha * fd1(:)));
    check(maxErr < 1e-4, sprintf('forward(%.1f*x) = %.1f*forward(x), maxErr=%.2e', ...
        alpha, alpha, maxErr));

    %% ---- 7. Channel independence ----
    fprintf('\n--- Channel independence ---\n');

    hopSize = 128;
    afSTFT(hopSize, 2, 2);
    rng(88);
    inFrame = [randn(hopSize * 10, 1), zeros(hopSize * 10, 1)];
    fd = afSTFT(inFrame);   % (bands x frames x channels)
    ch2_max = max(abs(fd(:, :, 2)), [], 'all');
    afSTFT();
    check(ch2_max < 1e-6, sprintf('silent ch2 max = %.2e', ch2_max));

    %% ---- 8. Band center frequencies ----
    fprintf('\n--- Band center frequencies ---\n');

    freqs = afSTFT(128, 1, 1);
    afSTFT();
    check(freqs(1) == 0,   sprintf('DC freq = %.4f (expect 0)', freqs(1)));
    check(freqs(end) == 1, sprintf('Nyquist freq = %.4f (expect 1)', freqs(end)));
    check(all(diff(freqs) > 0), 'frequencies monotonically increasing');

    freqs_h = afSTFT(128, 1, 1, 'hybrid');
    afSTFT();
    check(freqs_h(1) == 0,   sprintf('hybrid DC freq = %.4f', freqs_h(1)));
    check(freqs_h(end) == 1, sprintf('hybrid Nyquist freq = %.4f', freqs_h(end)));
    check(all(diff(freqs_h) > 0), 'hybrid frequencies monotonically increasing');

    %% ---- Summary ----
    fprintf('\n========================\n');
    fprintf('Results: %d passed, %d failed\n', passed, failed);
    if failed > 0
        error('test_afstft: %d test(s) FAILED', failed);
    end
end


%% ======== Helper: run continuous processing (pass-through) ========
function outSig = run_continuous(inSig, hopSize, fwdCh, invCh, opts)
    args = [{hopSize, fwdCh, invCh}, opts(:)'];
    afSTFT(args{:});

    sigLen = size(inSig, 1);
    nFrames = floor(sigLen / hopSize);
    outSig = zeros(nFrames * hopSize, invCh);

    for f = 1:nFrames
        idx = (f-1)*hopSize + (1:hopSize);
        fd = afSTFT(inSig(idx, :));
        td = afSTFT(fd);
        outSig(idx, :) = td;
    end
    afSTFT();
end


%% ======== Helper: find delay and compute SNR ========
function [snr, delay] = measure_snr(ref, test)
    ref  = ref(:);
    test = test(:);

    maxDelay = min(length(test), length(ref)) / 2;
    maxDelay = min(maxDelay, 20000);

    % Cross-correlation to find delay
    bestCorr = -inf;
    delay = 0;
    halfRef = floor(length(ref) / 2);
    for d = 0:maxDelay
        len = min(halfRef, length(test) - d);
        if len <= 0, break; end
        c = ref(1:len)' * test(d+1:d+len);
        if c > bestCorr
            bestCorr = c;
            delay = d;
        end
    end

    % Compute SNR over valid aligned region
    validLen = min(length(ref), length(test) - delay) - 500;
    if validLen < 1000
        snr = -inf;
        return;
    end
    aligned = test(delay+1 : delay+validLen);
    original = ref(1:validLen);

    noise = aligned - original;
    sigPow   = sum(original.^2);
    noisePow = sum(noise.^2);
    if noisePow < 1e-30
        snr = 999;
    else
        snr = 10 * log10(sigPow / noisePow);
    end
end
