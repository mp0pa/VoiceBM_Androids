import random
import csv
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import pandas as pd
from pydub import AudioSegment


def wav_to_mp3(folder_paths: list[str], output_dir: str = "mp3_converted", bitrate: str = "128k") -> list[str]:
    """
    Convert every .wav file found in a list of folders to .mp3 files stored in a
    dedicated output folder.

    Args:
        folder_paths: Paths to folders containing .wav files to convert.
        output_dir: Path to the folder where .mp3 files will be saved.
        bitrate: MP3 compression bitrate (e.g. "64k", "128k", "192k", "320k").

    Returns:
        List of paths to the generated .mp3 files.
    """
    dst_dir = Path(output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []

    for folder_path in folder_paths:
        src_dir = Path(folder_path)
        if not src_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {src_dir}")

        wav_files = list(src_dir.glob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"No .wav files found in: {src_dir}")

        for src in wav_files:
            dst = dst_dir / src.with_suffix(".mp3").name
            audio = AudioSegment.from_wav(str(src))
            audio.export(str(dst), format="mp3", bitrate=bitrate)
            output_paths.append(str(dst))

    return output_paths


def halve_csv_files(folder_paths: list[str], seed: Optional[int] = None) -> list[str]:
    """
    For each folder, randomly remove half the data rows from every .csv file
    and write the results to a new sibling folder named <folder>_halved.
    Original files are never modified.

    Args:
        folder_paths: Paths to folders containing .csv files to process.
        seed: Optional random seed for reproducibility.

    Returns:
        List of paths to the created output folders.
    """
    rng = random.Random(seed)
    output_folders = []

    for folder_path in folder_paths:
        src_dir = Path(folder_path)
        if not src_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {src_dir}")

        dst_dir = src_dir.parent / f"{src_dir.name}_halved"
        dst_dir.mkdir(parents=True, exist_ok=True)

        csv_files = list(src_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No .csv files found in: {src_dir}")

        for csv_file in csv_files:
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                rows = list(reader)

            keep_count = max(1, len(rows) // 2)
            kept_rows = rng.sample(rows, keep_count)

            dst_file = dst_dir / csv_file.name
            with open(dst_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if header is not None:
                    writer.writerow(header)
                writer.writerows(kept_rows)

        output_folders.append(str(dst_dir))

    return output_folders


def add_f0_stats_to_csv(
    folder_paths: list[str],
    csv_path: str,
    filename_col: str,
) -> str:
    """
    Extract F0 statistics from .mp3 files and add them as new columns to a CSV file.
    Uses librosa pyin — same approach as f0_extr_yin.py — which natively supports .mp3.

    For each row in the CSV, the matching .mp3 is found by comparing the value in
    `filename_col` (stem, without extension) against the stems of all .mp3 files
    found in `folder_paths`. This handles the case where the CSV still lists .wav
    filenames after audio has been converted to .mp3.

    Three new columns are added in place:
        F0_Mean_mp3, F0_StdDev_mp3, F0_Variance_mp3  (Hz, voiced frames only)

    Args:
        folder_paths:  Folders to search recursively for .mp3 files.
        csv_path:      Path to the CSV file to enrich (modified in place).
        filename_col:  Column in the CSV whose values identify the audio file
                       (matched by stem, extension-agnostic).

    Returns:
        Path to the updated CSV file.
    """
    # --- Collect all .mp3 files from every folder, keyed by stem ---
    mp3_index: dict[str, Path] = {}
    for folder_path in folder_paths:
        src_dir = Path(folder_path)
        if not src_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {src_dir}")
        for mp3_file in src_dir.rglob("*.mp3"):
            mp3_index[mp3_file.stem] = mp3_file

    # --- Load CSV ---
    df = pd.read_csv(csv_path)
    if filename_col not in df.columns:
        raise ValueError(f"Column '{filename_col}' not found in {csv_path}")

    f0_means, f0_stds, f0_vars = [], [], []

    for raw_value in df[filename_col]:
        # Match by stem so ".wav"/".mp3"/no-extension all resolve correctly
        stem = Path(str(raw_value)).stem
        mp3_path = mp3_index.get(stem)

        if mp3_path is None:
            print(f"  [Warning] No .mp3 found for '{stem}', filling with NaN.")
            f0_means.append(np.nan)
            f0_stds.append(np.nan)
            f0_vars.append(np.nan)
            continue

        try:
            # librosa.load natively supports .mp3 via ffmpeg — no conversion needed
            y, sr = librosa.load(str(mp3_path), sr=None)

            # 10 ms hop to stay consistent with f0_extr_yin.py
            hop_length = int(sr * 0.010)

            f0, _voiced_flag, _voiced_probs = librosa.pyin(
                y, fmin=50, fmax=500, sr=sr, hop_length=hop_length, center=False
            )

            # pyin marks unvoiced frames as NaN — compute stats on voiced frames only
            voiced_f0 = f0[~np.isnan(f0)]

            f0_means.append(voiced_f0.mean() if len(voiced_f0) > 0 else 0.0)
            f0_stds.append(voiced_f0.std()  if len(voiced_f0) > 1 else 0.0)
            f0_vars.append(voiced_f0.var()  if len(voiced_f0) > 1 else 0.0)

        except Exception as e:
            print(f"  [Error] Processing {mp3_path.name}: {e}")
            f0_means.append(np.nan)
            f0_stds.append(np.nan)
            f0_vars.append(np.nan)

    df["F0_Mean_mp3"]     = f0_means
    df["F0_StdDev_mp3"]   = f0_stds
    df["F0_Variance_mp3"] = f0_vars

    df.to_csv(csv_path, index=False)
    return csv_path


def add_halved_f0_to_csv(
    csv_path: str,
    folder_paths: list[str],
    filename_col: str,
) -> str:
    """
    Compute F0 mean, SD, and variance from per-utterance F0 CSV files and add
    them as new columns to the main CSV file.

    The folders should contain per-utterance CSV files with frame-level F0 values
    (e.g. the "values" output of the extraction scripts: Start_Time, End_Time, F0_Hz).
    For each such file the function:
      1. Reads the frame-level F0 values.
      2. Keeps only voiced frames (F0_Hz > 0).
      3. Computes mean, standard deviation, and variance.
      4. Writes the stats into the main CSV row whose `filename_col` value
         matches the per-utterance CSV filename stem (extension-agnostic).

    New columns added (parallel to add_f0_stats_to_csv, but sourced from
    pre-computed halved CSV files rather than raw .mp3 audio):
        F0_Mean_halved, F0_StdDev_halved, F0_Variance_halved

    Args:
        csv_path:      Main CSV file to enrich (modified in place).
        folder_paths:  Folders containing per-utterance F0 CSV files.
        filename_col:  Column in the main CSV that identifies each utterance
                       (matched by stem, extension-agnostic).

    Returns:
        Path to the updated CSV file.
    """
    # --- Index all per-utterance F0 CSV files by stem (mirrors mp3_index in add_f0_stats_to_csv) ---
    csv_index: dict[str, Path] = {}
    for folder_path in folder_paths:
        src_dir = Path(folder_path)
        if not src_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {src_dir}")
        for f0_csv in src_dir.rglob("*.csv"):
            csv_index[f0_csv.stem] = f0_csv

    # --- Load main CSV ---
    main_df = pd.read_csv(csv_path)
    if filename_col not in main_df.columns:
        raise ValueError(f"Column '{filename_col}' not found in {csv_path}.")

    means, stds, vars_ = [], [], []

    for raw_value in main_df[filename_col]:
        stem = Path(str(raw_value)).stem
        f0_csv_path = csv_index.get(stem)

        if f0_csv_path is None:
            print(f"  [Warning] No halved F0 CSV found for '{stem}', filling with NaN.")
            means.append(np.nan)
            stds.append(np.nan)
            vars_.append(np.nan)
            continue

        try:
            df_f0 = pd.read_csv(f0_csv_path)

            f0_col_candidates = [c for c in df_f0.columns if c.startswith("F0_")]
            if not f0_col_candidates:
                raise ValueError(f"No F0 column found in {f0_csv_path.name}.")
            f0_col = f0_col_candidates[0]

            voiced = df_f0[df_f0[f0_col] > 0][f0_col]

            means.append(voiced.mean() if len(voiced) > 0 else 0.0)
            stds.append(voiced.std()   if len(voiced) > 1 else 0.0)
            vars_.append(voiced.var()  if len(voiced) > 1 else 0.0)

        except Exception as e:
            print(f"  [Error] Processing {f0_csv_path.name}: {e}")
            means.append(np.nan)
            stds.append(np.nan)
            vars_.append(np.nan)

    main_df["F0_Mean_halved"]     = means
    main_df["F0_StdDev_halved"]   = stds
    main_df["F0_Variance_halved"] = vars_

    main_df.to_csv(csv_path, index=False)
    return csv_path


def _build_f0_index(folder_paths: list[str]) -> dict[str, tuple]:
    """
    Walk a list of F0 extraction output folders and build a stem → (mean, std, var)
    lookup.

    Expected structure (output of extract_f0 scripts):
        <folder>/
          <speaker_dir>/          ← "master folder", identifies the speaker
            mean/        <speaker>_mean.csv        → Utterance_File, F0_Mean
            standard_dev/<speaker>_standard_dev.csv → Utterance_File, F0_StdDev
            variance/    <speaker>_variance.csv     → Utterance_File, F0_Variance

    The utterance stem (e.g. "spk01_utt003" from "spk01_utt003.wav") is used as key.
    """
    index: dict[str, tuple] = {}

    for folder_path in folder_paths:
        src_dir = Path(folder_path)
        if not src_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {src_dir}")

        for speaker_dir in sorted(src_dir.iterdir()):
            if not speaker_dir.is_dir():
                continue

            mean_dir = speaker_dir / "mean"
            std_dir  = speaker_dir / "standard_dev"
            var_dir  = speaker_dir / "variance"

            if not all(d.is_dir() for d in [mean_dir, std_dir, var_dir]):
                continue

            mean_csvs = list(mean_dir.glob("*.csv"))
            std_csvs  = list(std_dir.glob("*.csv"))
            var_csvs  = list(var_dir.glob("*.csv"))

            if not mean_csvs or not std_csvs or not var_csvs:
                continue

            df_mean = pd.read_csv(mean_csvs[0])
            df_std  = pd.read_csv(std_csvs[0])
            df_var  = pd.read_csv(var_csvs[0])

            df = df_mean.merge(df_std, on="Utterance_File").merge(df_var, on="Utterance_File")

            for _, row in df.iterrows():
                stem = Path(str(row["Utterance_File"])).stem
                index[stem] = (row["F0_Mean"], row["F0_StdDev"], row["F0_Variance"])

    return index


def add_noisy_denoised_f0_to_csv(
    csv_path: str,
    noisy_folder_paths: list[str],
    denoised_folder_paths: list[str],
    filename_col: str,
) -> str:
    """
    Add pre-computed F0 mean, SD, and variance from noisy and denoised extraction
    output folders into the main CSV file.

    For each condition (noisy / denoised), the function searches speaker subfolders
    inside each provided folder path, reads the pre-computed stats from the "mean",
    "standard_dev", and "variance" subdirectories, and matches each utterance to
    the correct row in the main CSV via `filename_col` (extension-agnostic stem match).

    Six new columns are added:
        F0_Mean_noisy,    F0_StdDev_noisy,    F0_Variance_noisy
        F0_Mean_denoised, F0_StdDev_denoised, F0_Variance_denoised

    Rows with no matching utterance in the extraction output receive NaN.

    Args:
        csv_path:               Main CSV file to enrich (modified in place).
        noisy_folder_paths:     F0 extraction output folders for noisy data.
        denoised_folder_paths:  F0 extraction output folders for denoised data.
        filename_col:           Column in the main CSV identifying each utterance.

    Returns:
        Path to the updated CSV file.
    """
    noisy_index    = _build_f0_index(noisy_folder_paths)
    denoised_index = _build_f0_index(denoised_folder_paths)

    main_df = pd.read_csv(csv_path)
    if filename_col not in main_df.columns:
        raise ValueError(f"Column '{filename_col}' not found in {csv_path}.")

    noisy_means,    noisy_stds,    noisy_vars    = [], [], []
    denoised_means, denoised_stds, denoised_vars = [], [], []

    for raw_value in main_df[filename_col]:
        stem = Path(str(raw_value)).stem

        if stem in noisy_index:
            nm, ns, nv = noisy_index[stem]
        else:
            print(f"  [Warning] No noisy F0 found for '{stem}', filling with NaN.")
            nm, ns, nv = np.nan, np.nan, np.nan
        noisy_means.append(nm)
        noisy_stds.append(ns)
        noisy_vars.append(nv)

        if stem in denoised_index:
            dm, ds, dv = denoised_index[stem]
        else:
            print(f"  [Warning] No denoised F0 found for '{stem}', filling with NaN.")
            dm, ds, dv = np.nan, np.nan, np.nan
        denoised_means.append(dm)
        denoised_stds.append(ds)
        denoised_vars.append(dv)

    main_df["F0_Mean_noisy"]       = noisy_means
    main_df["F0_StdDev_noisy"]     = noisy_stds
    main_df["F0_Variance_noisy"]   = noisy_vars
    main_df["F0_Mean_denoised"]    = denoised_means
    main_df["F0_StdDev_denoised"]  = denoised_stds
    main_df["F0_Variance_denoised"] = denoised_vars

    main_df.to_csv(csv_path, index=False)
    return csv_path