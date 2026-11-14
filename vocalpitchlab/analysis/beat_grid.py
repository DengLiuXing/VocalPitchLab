"""曲速与逐拍检测：自相关定 BPM，再定位拍点与每拍强弱度。"""
from pathlib import Path

import numpy as np

BEATS_VERSION = 'beats-v1'
TARGET_RATE = 22050
N_FFT = 1024
HOP = 256
BLOCK_FRAMES = 512
BPM_MIN = 40.
BPM_MAX = 240.
BPM_STEP = .05
PRIOR_CENTER = 120.
PRIOR_OCTAVES = .9
HARMONICS = (1, 2, 3, 4)
MIN_BEATS = 8
UNSTABLE_RATIO = .15
LOW_CHANNELS = slice(0, 4)
LOW_WEIGHT = .6


def onset_envelope(samples, rate):
    """Positive spectral flux at the given rate; returns (envelope, frames per second)."""
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim > 1:
        mono = mono.mean(axis=0)
    pad = N_FFT // 2
    padded = np.pad(mono, (pad, pad)).astype(np.float32)
    count = 1 + (padded.size - N_FFT) // HOP
    if count < 2:
        return np.zeros(0, np.float32), rate / HOP
    window = np.hanning(N_FFT).astype(np.float32)
    offsets = np.arange(N_FFT)
    envelope = np.zeros(count, np.float32)
    previous = None
    # Whole-song rfft would allocate hundreds of MB on long inputs.
    for start in range(0, count, BLOCK_FRAMES):
        stop = min(count, start + BLOCK_FRAMES)
        index = offsets[None, :] + HOP * np.arange(start, stop)[:, None]
        magnitude = np.abs(np.fft.rfft(padded[index] * window, axis=1)).astype(np.float32)
        head = np.zeros((1, magnitude.shape[1]), np.float32) if previous is None else previous
        magnitude = np.vstack([head, magnitude])
        previous = magnitude[-1:]
        envelope[start:stop] = np.maximum(0., magnitude[1:] - magnitude[:-1]).sum(axis=1)
    return envelope, rate / HOP


def normalized_acf(envelope):
    """Mean-removed autocorrelation, normalized by overlap and by acf[0]."""
    if envelope.size < 2:
        return None
    centered = envelope - envelope.mean()
    size = 1
    while size < centered.size * 2:
        size <<= 1
    spectrum = np.fft.rfft(centered, size)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), size)[:centered.size]
    acf = acf / np.maximum(centered.size - np.arange(centered.size), 1)
    if not np.isfinite(acf[0]) or acf[0] <= 0:
        return None
    return acf / acf[0]


def estimate_bpm(envelope, fps):
    """Harmonic-weighted ACF peak with a log-normal prior; None when no beat is visible."""
    acf = normalized_acf(envelope)
    if acf is None:
        return None
    lags = np.arange(acf.size)
    candidates = np.arange(BPM_MIN, BPM_MAX + BPM_STEP * .5, BPM_STEP)
    score = np.zeros_like(candidates)
    for harmonic in HARMONICS:
        tap = harmonic * 60. * fps / candidates
        valid = tap < acf.size - 1
        if valid.any():
            score[valid] += np.interp(tap[valid], lags, acf) / harmonic
    # The prior is what separates 60/120/240 from each other; ACF alone cannot.
    score *= np.exp(-.5 * (np.log2(candidates / PRIOR_CENTER) / PRIOR_OCTAVES) ** 2)
    return round(float(candidates[int(np.argmax(score))]), 1)


def track_beats(envelope, fps, bpm):
    """Beat positions from a fixed tempo prior; the DP only resolves phase."""
    import librosa
    hop = int(round(TARGET_RATE / fps))
    _, beats = librosa.beat.beat_track(onset_envelope=envelope, sr=TARGET_RATE, hop_length=hop,
                                       bpm=bpm, units='time', trim=True)
    return np.asarray(beats, float)


def accent(mono, beats):
    """Per-beat attack strength; low band weights kick/bass, full band separates 1/3 from 2/4."""
    import librosa
    # channels takes one slice per band; a bare slice is read as an index list.
    low, full = librosa.onset.onset_strength_multi(y=mono, sr=TARGET_RATE, hop_length=HOP,
                                                   channels=[LOW_CHANNELS, slice(None)])
    times = librosa.frames_to_time(np.arange(full.size), sr=TARGET_RATE, hop_length=HOP)
    if full.size != times.size or not beats.size or full.size < 2:
        return np.zeros(beats.size, float)
    period = float(np.median(np.diff(beats))) if beats.size > 1 else .5
    half = max(.05, period * .5)
    values = np.empty(beats.size, float)
    for i, beat in enumerate(beats):
        a, b = np.searchsorted(times, [beat - half, beat + half])
        if b <= a:
            values[i] = 0.
            continue
        values[i] = LOW_WEIGHT * float(low[a:b].max()) + (1. - LOW_WEIGHT) * float(full[a:b].max())
    peak = float(values.max()) if values.size else 0.
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros(beats.size, float)
    return np.clip(values / peak, 0., 1.)


def extend_beats(beats, strength, duration):
    """Pad trimmed head/tail beats with the median period so the intro keeps a grid."""
    if beats.size < 2 or duration <= 0:
        return beats, strength
    period = float(np.median(np.diff(beats)))
    if not np.isfinite(period) or period <= .05:
        return beats, strength
    head = []
    value = float(beats[0]) - period
    while value > 0 and len(head) < 4096:
        head.append(value)
        value -= period
    tail = []
    value = float(beats[-1]) + period
    while value < duration and len(tail) < 4096:
        tail.append(value)
        value += period
    times = np.r_[np.asarray(head[::-1], float), beats, np.asarray(tail, float)]
    values = np.r_[np.zeros(len(head), float), strength, np.zeros(len(tail), float)]
    return times, values


def detect(path, duration):
    """Analyze one audio file. Empty time array means "tempo known, beats unreliable"."""
    import librosa
    source = Path(path)
    mono, _ = librosa.load(source, sr=TARGET_RATE, mono=True)
    if mono.size < N_FFT * 4:
        return None
    envelope, fps = onset_envelope(mono, TARGET_RATE)
    bpm = estimate_bpm(envelope, fps)
    if bpm is None:
        return None
    beats = track_beats(envelope, fps, bpm)
    if beats.size < MIN_BEATS:
        return None
    gaps = np.diff(beats)
    if gaps.size and float(gaps.std() / max(gaps.mean(), 1e-6)) > UNSTABLE_RATIO:
        return dict(bpm=bpm, time=np.zeros(0, float), accent=np.zeros(0, float))
    beats = beats[(beats >= 0) & (beats <= max(duration, beats[-1]))]
    if beats.size < MIN_BEATS:
        return dict(bpm=bpm, time=np.zeros(0, float), accent=np.zeros(0, float))
    strength = accent(mono, beats)
    times, values = extend_beats(beats, strength, duration)
    return dict(bpm=bpm, time=times, accent=values)
