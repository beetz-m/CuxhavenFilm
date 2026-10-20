"""A procedural ambient score for the film (no third-party music): chord pads, a low drone, a soft pulse in the
production chapters and swells at the chapter cuts. Deterministic (seeded).

score(duration_s, cuts, pulse_spans) -> stereo float32 array at RATE; write_wav(path, audio).
"""

from __future__ import annotations

import wave

import numpy as np
from scipy.signal import lfilter

RATE = 48000
# D minor colour: Dm9, Bbmaj7, Fadd9, Cadd9/E (MIDI note numbers)
CHORDS = [[50, 57, 62, 65, 69, 76], [46, 53, 58, 62, 69, 74], [41, 53, 57, 60, 67, 72], [40, 52, 55, 60, 67, 74]]
CHORD_S = 9.0


def hz(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def lowpass(x, cutoff):
    """One-pole low-pass."""
    a = np.exp(-2 * np.pi * cutoff / RATE)
    return lfilter([1 - a], [1, -a], x).astype(np.float32)


def pad(duration, rng):
    n = int(duration * RATE)
    t = np.arange(n) / RATE
    out = np.zeros((n, 2), np.float32)
    seg = int(CHORD_S * RATE)
    fade = int(3.0 * RATE)
    for i in range(0, n, seg):
        chord = CHORDS[(i // seg) % len(CHORDS)]
        j1 = min(n, i + seg + fade)
        tt = t[i:j1] - t[i]
        env = np.minimum(1.0, tt / 3.0) * np.clip((t[i:j1][-1] - t[i:j1]) / 3.0 + 0.0, 0, 1) ** 0.5
        env = np.minimum(env, 1.0)
        for k, note in enumerate(chord):
            for ch, det in enumerate((-0.12, 0.12)):
                f = hz(note) * 2 ** (det / 12)
                phase = rng.uniform(0, 2 * np.pi)
                v = (np.sin(2 * np.pi * f * tt + phase) + 0.35 * np.sin(4 * np.pi * f * tt + phase)
                     + 0.12 * np.sin(6 * np.pi * f * tt))
                amp = 0.05 if note > 60 else 0.07
                out[i:j1, ch] += (amp * env * v).astype(np.float32)
    # slow filter sweep: warm, then a little brighter, then warm
    for ch in range(2):
        out[:, ch] = 0.6 * lowpass(out[:, ch], 900) + 0.4 * lowpass(out[:, ch], 2400)
    return out


def drone(duration):
    n = int(duration * RATE)
    t = np.arange(n) / RATE
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * t / 23.0)
    d = 0.09 * np.sin(2 * np.pi * hz(38) * t) * (0.7 + 0.3 * lfo) + 0.04 * np.sin(2 * np.pi * hz(26) * t)
    return np.stack([d, d], 1).astype(np.float32)


def pulse(duration, spans, bpm=84):
    n = int(duration * RATE)
    out = np.zeros(n, np.float32)
    beat = 60.0 / bpm
    hit_t = np.arange(int(0.45 * RATE)) / RATE
    hit = (np.sin(2 * np.pi * 55 * hit_t * (1 + 0.6 * np.exp(-hit_t * 30))) * np.exp(-hit_t * 9)).astype(np.float32)
    tick = (np.random.default_rng(3).normal(0, 1, int(0.05 * RATE)) * np.exp(-np.arange(int(0.05 * RATE)) / RATE * 90))
    tick = lowpass(tick.astype(np.float32), 5000) * 0.25
    for a, b in spans:
        k = 0
        x = a
        while x < b:
            fade = min(1.0, (x - a) / 4.0, (b - x) / 4.0)
            i = int(x * RATE)
            seg = hit[: max(0, min(len(hit), n - i))]
            out[i:i + len(seg)] += 0.22 * fade * seg
            if k % 2 == 1:
                j = int((x + beat / 2) * RATE)
                s2 = tick[: max(0, min(len(tick), n - j))]
                out[j:j + len(s2)] += fade * s2
            x += beat
            k += 1
    return np.stack([out, out], 1)


def swells(duration, cuts):
    n = int(duration * RATE)
    out = np.zeros((n, 2), np.float32)
    rng = np.random.default_rng(7)
    L = int(3.5 * RATE)
    noise = rng.normal(0, 1, (L, 2)).astype(np.float32)
    for ch in range(2):
        noise[:, ch] = lowpass(noise[:, ch], 1800)
    env = (np.linspace(0, 1, L) ** 3).astype(np.float32)
    for c in cuts:
        j = int(c * RATE) - L
        if j < 0:
            continue
        out[j:j + L] += 0.10 * noise * env[:, None]
    return out


def score(duration, cuts=(), pulse_spans=()):
    rng = np.random.default_rng(11)
    a = pad(duration, rng) + drone(duration) + pulse(duration, pulse_spans) + swells(duration, cuts)
    n = len(a)
    fi, fo = int(2.5 * RATE), int(5.0 * RATE)
    a[:fi] *= np.linspace(0, 1, fi)[:, None]
    a[-fo:] *= np.linspace(1, 0, fo)[:, None]
    peak = np.abs(a).max()
    return (a / peak * 0.6).astype(np.float32)


def write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "score_test.wav")
    write_wav(out, score(40, cuts=[20], pulse_spans=[(8, 30)]))
    print(out)
