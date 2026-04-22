#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compute audio quality metrics for each denoised file and write a CSV report.

For every WAV file present in out_dir the script:
  1. Loads the matching original from in_dir (reference) and the processed file
     (estimate).
  2. Computes three metrics against the reference:
       - PESQ   (Perceptual Evaluation of Speech Quality, wideband at 16 kHz)
       - SI-SAR (Scale-Invariant Signal-to-Artifacts Ratio)
       - STOI   (Short-Time Objective Intelligibility)
  3. Writes results to a CSV named <out_folder>_<YYYYMMDD>_<HHMMSS>.csv in out_dir.

Files are processed in parallel (one worker per CPU core).

Dependencies: pesq, pystoi, soundfile, numpy, scipy
"""

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from math import gcd

import numpy as np
import scipy.signal
import soundfile as sf
from pesq import pesq
from pystoi import stoi

# ---------------------------------------------------------------------------
# Same paths as denoisepipe.py
# ---------------------------------------------------------------------------
in_dir = "/home/mae/Documents/idmc/master1/university/s2/supervised_project/corpus/Androids-Corpus/Androids-Corpus/Interview-Task/audio/HC"
out_dir = "/home/mae/Documents/idmc/master1/university/s2/supervised_project/corpus/Androids-Corpus/Androids-Corpus/Interview-Task/audio/HC_cleaned"

# PESQ requires exactly 8 000 Hz (narrowband) or 16 000 Hz (wideband).
PESQ_SR = 16_000
# Chunk length fed to the pesq C extension (crashes on buffers > ~10 s).
PESQ_CHUNK_SEC = 8.0
# Minimum chunk length accepted by pesq.
PESQ_MIN_SEC = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample (faster than FFT-based resample for integer ratios)."""
    if orig_sr == target_sr:
        return audio
    g = gcd(orig_sr, target_sr)
    return scipy.signal.resample_poly(audio, target_sr // g, orig_sr // g)


# ---------------------------------------------------------------------------
# Metric implementations
# ---------------------------------------------------------------------------

def compute_pesq(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    """PESQ wideband score (MOS-LQO ≈ -0.5 – 4.5), averaged over 8-second chunks.

    The pesq C extension has fixed-size internal buffers and crashes (stack
    smashing) on audio longer than ~10 s, so input is split into chunks.
    """
    ref_r = resample(ref, sr, PESQ_SR)
    est_r = resample(est, sr, PESQ_SR)
    n = min(len(ref_r), len(est_r))
    ref_r, est_r = ref_r[:n], est_r[:n]

    chunk = int(PESQ_CHUNK_SEC * PESQ_SR)
    min_chunk = int(PESQ_MIN_SEC * PESQ_SR)
    scores = [
        pesq(PESQ_SR, ref_r[s:s + chunk], est_r[s:s + chunk], "wb")
        for s in range(0, n, chunk)
        if n - s >= min_chunk
    ]
    return float(np.mean(scores))


def compute_stoi(ref: np.ndarray, est: np.ndarray, sr: int) -> float:
    """STOI score (0 – 1; higher = more intelligible)."""
    n = min(len(ref), len(est))
    return float(stoi(ref[:n], est[:n], sr, extended=False))


def compute_si_sar(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-Invariant Signal-to-Artifacts Ratio (dB).

    Avoids materialising intermediate arrays via the identity:
        ||s_target||²  =  dot(est, ref)² / ||ref||²
        ||e_artif||²   =  ||est||²  −  ||s_target||²
    """
    n = min(len(ref), len(est))
    ref = ref[:n] - ref[:n].mean()
    est = est[:n] - est[:n].mean()

    ref_energy = np.dot(ref, ref)
    if ref_energy < 1e-10:
        return float("nan")

    s_target_energy = np.dot(est, ref) ** 2 / ref_energy
    artif_energy = np.dot(est, est) - s_target_energy
    if artif_energy < 1e-10:
        return float("inf")

    return float(10.0 * np.log10(s_target_energy / artif_energy))


# ---------------------------------------------------------------------------
# Per-file computation (runs in worker process)
# ---------------------------------------------------------------------------

def compute_metrics(filename: str) -> dict:
    """Load one pair of files, compute all metrics, return a result dict."""
    ref_path = os.path.join(in_dir, filename)
    est_path = os.path.join(out_dir, filename)

    ref_data, ref_sr = sf.read(ref_path, dtype="float32")
    est_data, est_sr = sf.read(est_path, dtype="float32")

    ref = to_mono(ref_data)
    est = to_mono(est_data)

    if est_sr != ref_sr:
        est = resample(est, est_sr, ref_sr)

    return {
        "file":   filename,
        "PESQ":   round(compute_pesq(ref, est, ref_sr), 4),
        "SI-SAR": round(compute_si_sar(ref, est), 4),
        "STOI":   round(compute_stoi(ref, est, ref_sr), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    wav_files = sorted(f for f in os.listdir(out_dir) if f.lower().endswith(".wav"))
    wav_files = [f for f in wav_files if os.path.exists(os.path.join(in_dir, f))]

    if not wav_files:
        print("No matching WAV files found.")
        return

    skipped = set(os.listdir(out_dir)) - set(wav_files)
    for f in skipped:
        if f.lower().endswith(".wav"):
            print(f"  [skip] No matching reference for {f}")

    folder_name = os.path.basename(os.path.normpath(out_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(out_dir, f"{folder_name}_{timestamp}.csv")

    futures = {}
    with (
        ProcessPoolExecutor() as pool,
        open(output_csv, "w", newline="") as fh,
    ):
        writer = csv.DictWriter(fh, fieldnames=["file", "PESQ", "SI-SAR", "STOI"])
        writer.writeheader()

        for f in wav_files:
            futures[pool.submit(compute_metrics, f)] = f

        for future in as_completed(futures):
            filename = futures[future]
            try:
                row = future.result()
                writer.writerow(row)
                fh.flush()
                print(f"  {filename}  PESQ={row['PESQ']:.4f}  SI-SAR={row['SI-SAR']:.4f}  STOI={row['STOI']:.4f}")
            except Exception as exc:
                writer.writerow({"file": filename, "PESQ": "error", "SI-SAR": "error", "STOI": "error"})
                fh.flush()
                print(f"  [error] {filename}: {exc}")

    print(f"\nMetrics saved to {output_csv}")


if __name__ == "__main__":
    main()