import random
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import pandas as pd
from pydub import AudioSegment

_F0_DTYPE = {"F0_Mean": "float64", "F0_StdDev": "float64", "F0_Variance": "float64"}


def wav_to_mp3(folder_paths: list[str], output_dir: str = "mp3_converted", bitrate: str = "128k") -> list[str]:
    """
    Convert every .wav file found in a list of folders to .mp3 files stored in a
    dedicated output folder.

    Args:
        folder_paths: Mother folders to search recursively for .wav files.
                      Each folder is walked and every .wav found at any depth
                      is converted (e.g. provide the segmented speaker parent
                      folder and all speaker sub-folders are processed).
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

        wav_files = list(src_dir.rglob("*.wav"))
        if not wav_files:
            raise FileNotFoundError(f"No .wav files found under: {src_dir}")

        for src in wav_files:
            dst = dst_dir / src.with_suffix(".mp3").name
            AudioSegment.from_wav(str(src)).export(str(dst), format="mp3", bitrate=bitrate)
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

            kept_rows = rng.sample(rows, max(1, len(rows) // 2))

            dst_file = dst_dir / csv_file.name
            with open(dst_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if header is not None:
                    writer.writerow(header)
                writer.writerows(kept_rows)

        output_folders.append(str(dst_dir))

    return output_folders


# Base (reference) level for each condition column.
_BASE_CONDITIONS: dict[str, str] = {
    "toolbox":      "librosa",
    "noise":        "noisy",
    "compression":  "wav",
    "dataset_size": "full",
}

# Suffixes that identify already-expanded condition rows; used to prevent
# each add_* step from compounding conditions across steps.
_COND_SUFFIXES: tuple[str, ...] = (
    "_compression_", "_halving_", "_noise_", "_toolbox_"
)


def _conditions_from_utterance_id(utterance_id: str) -> dict[str, str]:
    """
    Parse condition values encoded in an utterance_id string.

    For any condition not found in the id, the base (reference) level from
    _BASE_CONDITIONS is returned.  The function is used by every add_* pipeline
    step to ensure all condition columns are always fully populated.

    Encoding conventions (set by the add_* functions themselves):
        compression:  _compression_mp3  |  _compression_wav
        dataset_size: _halving_halved   |  _halving_full
        noise:        _noise_noisy      |  _noise_denoised
        toolbox:      _toolbox_<name>
    """
    conditions = dict(_BASE_CONDITIONS)

    if "_compression_mp3" in utterance_id:
        conditions["compression"] = "mp3"
    elif "_compression_wav" in utterance_id:
        conditions["compression"] = "wav"

    if "_halving_halved" in utterance_id:
        conditions["dataset_size"] = "halved"
    elif "_halving_full" in utterance_id:
        conditions["dataset_size"] = "full"

    if "_noise_noisy" in utterance_id:
        conditions["noise"] = "noisy"
    elif "_noise_denoised" in utterance_id:
        conditions["noise"] = "denoised"

    if "_toolbox_" in utterance_id:
        conditions["toolbox"] = utterance_id.split("_toolbox_", 1)[1].split("_")[0]

    return conditions


def _build_f0_index(folder_paths: list[str]) -> dict[str, tuple]:
    """
    Walk a list of F0 extraction output folders and build a speaker_id → (mean, std, var)
    lookup, aggregating all utterances that belong to the same speaker.

    Expected structure (output of extract_f0 scripts):
        <folder>/
          <speaker_dir>/
            mean/        <speaker>_mean.csv        → Utterance_File, F0_Mean
            standard_dev/<speaker>_standard_dev.csv → Utterance_File, F0_StdDev
            variance/    <speaker>_variance.csv     → Utterance_File, F0_Variance

    The speaker ID (e.g. "IT_01_CF56_1" stripped of "_utt###") is used as key.
    Aggregation: F0_Mean = mean of utterance means;
                 F0_Variance = mean of utterance variances + variance of utterance means
                 (law of total variance); F0_StdDev = sqrt(F0_Variance).
    """
    speaker_data: dict[str, list[tuple[float, float, float]]] = {}

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

            if not all(d.is_dir() for d in (mean_dir, std_dir, var_dir)):
                continue

            mean_csvs = list(mean_dir.glob("*.csv"))
            std_csvs  = list(std_dir.glob("*.csv"))
            var_csvs  = list(var_dir.glob("*.csv"))

            if not mean_csvs or not std_csvs or not var_csvs:
                continue

            df = (
                pd.read_csv(mean_csvs[0])
                  .merge(pd.read_csv(std_csvs[0]), on="Utterance_File")
                  .merge(pd.read_csv(var_csvs[0]), on="Utterance_File")
            )

            for utt, mean, std, var in zip(
                df["Utterance_File"], df["F0_Mean"], df["F0_StdDev"], df["F0_Variance"]
            ):
                spk = _speaker_id_from_stem(Path(str(utt)).stem)
                speaker_data.setdefault(spk, []).append((float(mean), float(std), float(var)))

    index: dict[str, tuple] = {}
    for spk, stats in speaker_data.items():
        means = np.array([s[0] for s in stats])
        vars_ = np.array([s[2] for s in stats])
        agg_mean = float(np.mean(means))
        agg_var  = float(np.mean(vars_) + np.var(means))
        agg_std  = float(np.sqrt(agg_var))
        index[spk] = (agg_mean, agg_std, agg_var)

    return index


def _extract_mp3_f0(mp3_path: Path) -> tuple[float, float, float]:
    y, sr = librosa.load(str(mp3_path), sr=None)
    f0, _, _ = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr,
        hop_length=int(sr * 0.010), center=False,
    )
    voiced = f0[~np.isnan(f0)]
    return (
        float(voiced.mean()) if len(voiced) > 0 else 0.0,
        float(voiced.std())  if len(voiced) > 1 else 0.0,
        float(voiced.var())  if len(voiced) > 1 else 0.0,
    )


def add_f0_stats_compression_to_csv(
    csv_path: str,
    mp3_folder_paths: list[str],
    f0_csv_folder_paths: list[str],
) -> str:
    """
    For each utterance in the main CSV, append two new rows:
      - One for the mp3 condition: F0 extracted live from .mp3 via librosa pyin.
      - One for the wav condition: F0 read from pre-computed per-utterance CSVs.

    The utterance_id in each appended row encodes the original filename, the modality
    (compression), and the compression level (mp3 or wav):
        <stem>_compression_mp3
        <stem>_compression_wav

    All existing columns are carried over from the source row.  Three columns are
    filled with condition-specific F0 values, and the "compression" column is set
    to "mp3" or "wav" depending on the condition.

    Args:
        csv_path:             Main CSV file to enrich (rows appended, saved in place).
        mp3_folder_paths:     Folders to search recursively for .mp3 files.
        f0_csv_folder_paths:  Mother folders each containing speaker sub-folders
                              that directly hold mean/, standard_dev/, and variance/
                              sub-folders with pre-computed per-utterance F0 stats
                              from .wav files (e.g. [".../F0_results_IT"]).

    Returns:
        Path to the updated CSV file.
    """
    # Index .mp3 files by speaker across all mp3 folders (recursive)
    mp3_speaker_index: dict[str, list[Path]] = {}
    for folder_path in mp3_folder_paths:
        mp3_dir = Path(folder_path)
        if not mp3_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {mp3_dir}")
        for p in mp3_dir.rglob("*.mp3"):
            spk = _speaker_id_from_stem(p.stem)
            mp3_speaker_index.setdefault(spk, []).append(p)

    wav_index = _build_f0_index(f0_csv_folder_paths)

    # Extract F0 stats for all mp3 utterances in parallel, then aggregate per speaker
    mp3_utt_cache: dict[str, tuple] = {}
    all_mp3_paths = [p for paths in mp3_speaker_index.values() for p in paths]
    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(_extract_mp3_f0, p): p for p in all_mp3_paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                mp3_utt_cache[path.stem] = future.result()
            except Exception as e:
                print(f"  [Error] Processing {path.name}: {e}")
                mp3_utt_cache[path.stem] = (np.nan, np.nan, np.nan)

    mp3_f0_cache: dict[str, tuple] = {}
    for spk, paths in mp3_speaker_index.items():
        utt_stats = [mp3_utt_cache[p.stem] for p in paths if p.stem in mp3_utt_cache]
        valid = [(m, s, v) for m, s, v in utt_stats if not np.isnan(m)]
        if not valid:
            mp3_f0_cache[spk] = (np.nan, np.nan, np.nan)
        else:
            means = np.array([s[0] for s in valid])
            vars_ = np.array([s[2] for s in valid])
            agg_mean = float(np.mean(means))
            agg_var  = float(np.mean(vars_) + np.var(means))
            mp3_f0_cache[spk] = (agg_mean, float(np.sqrt(agg_var)), agg_var)

    main_df = pd.read_csv(csv_path, dtype=_F0_DTYPE)
    new_rows = []

    base_mask = ~main_df["Utterance_File"].astype(str).apply(
        lambda x: any(s in x for s in _COND_SUFFIXES)
    )
    for source_record in main_df[base_mask].to_dict("records"):
        stem = Path(str(source_record["Utterance_File"])).stem
        base_cond = _conditions_from_utterance_id(str(source_record["Utterance_File"]))

        # mp3 row
        mp3_record = {**source_record, "Utterance_File": f"{stem}_compression_mp3",
                      **base_cond, "compression": "mp3"}
        if stem in mp3_f0_cache:
            mp3_record["F0_Mean"], mp3_record["F0_StdDev"], mp3_record["F0_Variance"] = mp3_f0_cache[stem]
        else:
            print(f"  [Warning] No .mp3 found for speaker '{stem}', filling F0 with NaN.")
            mp3_record["F0_Mean"] = mp3_record["F0_StdDev"] = mp3_record["F0_Variance"] = np.nan
        new_rows.append(mp3_record)

        # wav row
        wav_record = {**source_record, "Utterance_File": f"{stem}_compression_wav",
                      **base_cond, "compression": "wav"}
        if stem in wav_index:
            wav_record["F0_Mean"], wav_record["F0_StdDev"], wav_record["F0_Variance"] = wav_index[stem]
        else:
            print(f"  [Warning] No wav F0 stats found for speaker '{stem}', filling with NaN.")
            wav_record["F0_Mean"] = wav_record["F0_StdDev"] = wav_record["F0_Variance"] = np.nan
        new_rows.append(wav_record)

    result_df = pd.concat([main_df, pd.DataFrame(new_rows)], ignore_index=True)
    result_df.to_csv(csv_path, index=False)
    return csv_path


def add_halved_f0_to_csv(
    csv_path: str,
    halved_folder_paths: list[str],
    f0_csv_folder_paths: list[str],
) -> str:
    """
    For each utterance in the main CSV, append two new rows:
      - One for the halved condition: F0 computed from per-utterance frame-level
        CSVs in `halved_folder_paths` (voiced frames where F0_Hz > 0).
      - One for the full condition: F0 read from pre-computed stats in the
        mean/, standard_dev/, and variance/ sub-folders of `f0_csv_folder_paths`.

    The utterance_id in each appended row encodes the original filename, the
    modality (halving), and the level (halved or full):
        <stem>_halving_halved
        <stem>_halving_full

    All existing columns are carried over from the source row.  F0_Mean,
    F0_StdDev, and F0_Variance are filled with condition-specific values, and
    the "dataset_size" column is set to "halved" or "full".

    Args:
        csv_path:             Main CSV file to enrich (rows appended, saved in place).
        halved_folder_paths:  Folders to search recursively for per-utterance
                              frame-level F0 CSVs from the halved data
                              (Start_Time, End_Time, F0_Hz).
        f0_csv_folder_paths:  Mother folders each containing speaker sub-folders
                              that directly hold mean/, standard_dev/, and variance/
                              sub-folders with pre-computed F0 stats from the full data
                              (e.g. [".../F0_results_IT"]).

    Returns:
        Path to the updated CSV file.
    """
    # Index halved frame-level CSVs by speaker (recursive), grouping all utterances
    halved_speaker_index: dict[str, list[Path]] = {}
    for folder_path in halved_folder_paths:
        src_dir = Path(folder_path)
        if not src_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {src_dir}")
        for f0_csv in src_dir.rglob("*.csv"):
            spk = _speaker_id_from_stem(f0_csv.stem)
            halved_speaker_index.setdefault(spk, []).append(f0_csv)

    full_index = _build_f0_index(f0_csv_folder_paths)
    main_df = pd.read_csv(csv_path, dtype=_F0_DTYPE)
    new_rows = []

    base_mask = ~main_df["Utterance_File"].astype(str).apply(
        lambda x: any(s in x for s in _COND_SUFFIXES)
    )
    for source_record in main_df[base_mask].to_dict("records"):
        stem = Path(str(source_record["Utterance_File"])).stem
        base_cond = _conditions_from_utterance_id(str(source_record["Utterance_File"]))

        # halved row — pool all voiced frames across utterances for this speaker
        halved_record = {**source_record, "Utterance_File": f"{stem}_halving_halved",
                         **base_cond, "dataset_size": "halved"}
        csv_paths = halved_speaker_index.get(stem)
        if csv_paths is None:
            print(f"  [Warning] No halved F0 CSV found for speaker '{stem}', filling with NaN.")
            halved_record["F0_Mean"] = halved_record["F0_StdDev"] = halved_record["F0_Variance"] = np.nan
        else:
            try:
                voiced_frames = []
                for f0_csv_path in csv_paths:
                    df_f0 = pd.read_csv(f0_csv_path)
                    f0_col_candidates = [c for c in df_f0.columns if c.startswith("F0_")]
                    if not f0_col_candidates:
                        raise ValueError(f"No F0 column found in {f0_csv_path.name}.")
                    voiced_frames.append(df_f0[df_f0[f0_col_candidates[0]] > 0][f0_col_candidates[0]])
                voiced = pd.concat(voiced_frames) if voiced_frames else pd.Series([], dtype=float)
                halved_record["F0_Mean"]     = float(voiced.mean()) if len(voiced) > 0 else 0.0
                halved_record["F0_StdDev"]   = float(voiced.std())  if len(voiced) > 1 else 0.0
                halved_record["F0_Variance"] = float(voiced.var())  if len(voiced) > 1 else 0.0
            except Exception as e:
                print(f"  [Error] Processing halved CSVs for '{stem}': {e}")
                halved_record["F0_Mean"] = halved_record["F0_StdDev"] = halved_record["F0_Variance"] = np.nan
        new_rows.append(halved_record)

        # full row
        full_record = {**source_record, "Utterance_File": f"{stem}_halving_full",
                       **base_cond, "dataset_size": "full"}
        if stem in full_index:
            full_record["F0_Mean"], full_record["F0_StdDev"], full_record["F0_Variance"] = full_index[stem]
        else:
            print(f"  [Warning] No full F0 stats found for '{stem}', filling with NaN.")
            full_record["F0_Mean"] = full_record["F0_StdDev"] = full_record["F0_Variance"] = np.nan
        new_rows.append(full_record)

    result_df = pd.concat([main_df, pd.DataFrame(new_rows)], ignore_index=True)
    result_df.to_csv(csv_path, index=False)
    return csv_path


def add_noisy_denoised_f0_to_csv(
    csv_path: str,
    noisy_folder_paths: list[str],
    denoised_folder_paths: list[str],
) -> str:
    """
    For each utterance in the main CSV, append two new rows:
      - One for the noisy condition: F0 read from pre-computed stats in
        `noisy_folder_paths` (speaker subfolders with mean/, standard_dev/,
        variance/ sub-folders).
      - One for the denoised condition: same structure in `denoised_folder_paths`.

    The utterance_id in each appended row encodes the original filename, the
    modality (noise), and the level (noisy or denoised):
        <stem>_noise_noisy
        <stem>_noise_denoised

    All existing columns are carried over from the source row.  F0_Mean,
    F0_StdDev, and F0_Variance are filled with condition-specific values, and
    the "noise" column is set to "noisy" or "denoised".

    Args:
        csv_path:               Main CSV file to enrich (rows appended, saved in place).
        noisy_folder_paths:     F0 extraction output folders for the noisy condition.
        denoised_folder_paths:  F0 extraction output folders for the denoised condition.

    Returns:
        Path to the updated CSV file.
    """
    noise_indices = {
        "noisy":    _build_f0_index(noisy_folder_paths),
        "denoised": _build_f0_index(denoised_folder_paths),
    }
    main_df = pd.read_csv(csv_path, dtype=_F0_DTYPE)
    new_rows = []

    base_mask = ~main_df["Utterance_File"].astype(str).apply(
        lambda x: any(s in x for s in _COND_SUFFIXES)
    )
    for source_record in main_df[base_mask].to_dict("records"):
        stem = Path(str(source_record["Utterance_File"])).stem
        base_cond = _conditions_from_utterance_id(str(source_record["Utterance_File"]))

        for label, index in noise_indices.items():
            record = {**source_record, "Utterance_File": f"{stem}_noise_{label}",
                      **base_cond, "noise": label}
            if stem in index:
                record["F0_Mean"], record["F0_StdDev"], record["F0_Variance"] = index[stem]
            else:
                print(f"  [Warning] No {label} F0 found for '{stem}', filling with NaN.")
                record["F0_Mean"] = record["F0_StdDev"] = record["F0_Variance"] = np.nan
            new_rows.append(record)

    result_df = pd.concat([main_df, pd.DataFrame(new_rows)], ignore_index=True)
    result_df.to_csv(csv_path, index=False)
    return csv_path


def add_toolbox_f0_to_csv(
    csv_path: str,
    **toolbox_folder_paths: list[str],
) -> str:
    """
    For each utterance in the main CSV, append one new row per toolbox.

    Toolboxes are passed as keyword arguments mapping a tool name to a list of
    folder paths containing its F0 extraction output (speaker subfolders with
    mean/, standard_dev/, variance/ sub-folders):

        add_toolbox_f0_to_csv(
            csv_path,
            egemaps=["path/to/egemaps/results"],
            praat=["path/to/praat/results"],
            librosa=["path/to/librosa/results"],
        )

    The utterance_id in each appended row encodes the original filename, the
    modality (toolbox), and the tool name:
        <stem>_toolbox_egemaps
        <stem>_toolbox_praat
        <stem>_toolbox_librosa

    All existing columns are carried over from the source row.  F0_Mean,
    F0_StdDev, and F0_Variance are filled with tool-specific values, and the
    "toolbox" column is set to the tool name.

    Args:
        csv_path:               Main CSV file to enrich (rows appended, saved in place).
        **toolbox_folder_paths: Keyword arguments where each key is a tool name and
                                each value is a list of folder paths for that tool's
                                F0 extraction output.

    Returns:
        Path to the updated CSV file.
    """
    toolbox_indices: dict[str, dict[str, tuple]] = {
        tool: _build_f0_index(folder_paths)
        for tool, folder_paths in toolbox_folder_paths.items()
    }
    main_df = pd.read_csv(csv_path, dtype=_F0_DTYPE)
    new_rows = []

    base_mask = ~main_df["Utterance_File"].astype(str).apply(
        lambda x: any(s in x for s in _COND_SUFFIXES)
    )
    for source_record in main_df[base_mask].to_dict("records"):
        stem = Path(str(source_record["Utterance_File"])).stem
        base_cond = _conditions_from_utterance_id(str(source_record["Utterance_File"]))

        for tool, index in toolbox_indices.items():
            record = {**source_record, "Utterance_File": f"{stem}_toolbox_{tool}", **base_cond, "toolbox": tool}
            if stem in index:
                record["F0_Mean"], record["F0_StdDev"], record["F0_Variance"] = index[stem]
            else:
                print(f"  [Warning] No {tool} F0 found for speaker '{stem}', filling with NaN.")
                record["F0_Mean"] = record["F0_StdDev"] = record["F0_Variance"] = np.nan
            new_rows.append(record)

    result_df = pd.concat([main_df, pd.DataFrame(new_rows)], ignore_index=True)
    result_df.to_csv(csv_path, index=False)
    return csv_path


def _speaker_id_from_stem(stem: str) -> str:
    """Strip the _utt### suffix (and optional trailing 'ok') to get the speaker identifier."""
    idx = stem.find("_utt")
    spk = stem[:idx] if idx != -1 else stem
    return spk.removesuffix("ok")


def _parse_depression_diagnosis(utterance_stem: str) -> str:
    """
    Extract the depression diagnosis label from an utterance filename stem.

    Expected format: TASK_PATIENTID_DIAGSEXAGE_EDUCATION_UTT
        e.g. IT_01_CF56_1_utt001
             ^^  ^^  ^^ ^^
             |   |   |  age (56)
             |   |   diagnosis + sex code: C/P + F/M  (e.g. CF = Control Female)
             |   patient id
             task id (IT or RT)

    The diagnosis+sex+age field has no internal separator: the first char is the
    diagnosis (C = control, P = patient), the second is the sex (F/M), and the
    remaining digits are the patient's age.

    Returns "C" (control), "P" (patient), or "unknown" if the stem cannot be parsed.
    """
    parts = utterance_stem.split("_")
    if len(parts) >= 3 and parts[2] and parts[2][0] in ("C", "P"):
        return parts[2][0]
    return "unknown"


def main(
    output_csv: str,
    wav_folder_paths: list[str],
    mp3_output_dir: str,
    f0_wav_folder_paths: list[str],
    f0_values_folder_paths: list[str],
    f0_full_folder_paths: list[str],
    noisy_folder_paths: list[str],
    denoised_folder_paths: list[str],
    toolbox_folder_paths: dict[str, list[str]],
    mp3_bitrate: str = "128k",
    halve_seed: Optional[int] = None,
) -> str:
    """
    Build the complete LMM verification CSV described in README_LMM.md.

    Pipeline:
      1. Convert .wav files to .mp3 (compression condition source).
      2. Randomly halve the frame-level F0 value CSVs (halving condition source).
      3. Create a template CSV with one row per utterance carrying only the
         utterance_id and depression_diagnosis (parsed from the filename).
      4. Append compression condition rows  (add_f0_stats_compression_to_csv).
      5. Append halving condition rows      (add_halved_f0_to_csv).
      6. Append noise condition rows        (add_noisy_denoised_f0_to_csv).
      7. Append toolbox condition rows      (add_toolbox_f0_to_csv).
      8. Drop the template rows (no condition assigned) and rename columns to
         the nine-column schema from the README.

    Args:
        output_csv:              Path to the final output CSV file.
        wav_folder_paths:        Folders containing the original .wav utterances.
        mp3_output_dir:          Destination folder for the converted .mp3 files.
        f0_wav_folder_paths:     Speaker folders (mean/standard_dev/variance) with
                                 pre-computed F0 stats from original .wav files,
                                 used as the wav side of the compression condition.
        f0_values_folder_paths:  Folders with frame-level F0 value CSVs to halve.
        f0_full_folder_paths:    Speaker folders (mean/standard_dev/variance) with
                                 pre-computed F0 stats from the full dataset, used
                                 as the full side of the halving condition.
        noisy_folder_paths:      Speaker folders (mean/standard_dev/variance) with
                                 pre-computed F0 stats from noisy audio.
        denoised_folder_paths:   Speaker folders (mean/standard_dev/variance) with
                                 pre-computed F0 stats from denoised audio.
        toolbox_folder_paths:    Mapping of tool name → speaker folders
                                 (mean/standard_dev/variance) for that tool's F0
                                 extraction output.
        mp3_bitrate:             MP3 bitrate for wav→mp3 conversion (default "128k").
        halve_seed:              Optional random seed for reproducible halving.

    Returns:
        Path to the completed CSV file.
    """
    # 1 — Convert wav to mp3
    wav_to_mp3(wav_folder_paths, mp3_output_dir, mp3_bitrate)

    # 2 — Discover values/ sub-folders inside each speaker dir, then halve their CSVs
    values_paths = [
        str(speaker_dir / "values")
        for mother_path in f0_values_folder_paths
        for speaker_dir in sorted(Path(mother_path).iterdir())
        if speaker_dir.is_dir() and (speaker_dir / "values").is_dir()
    ]
    halved_folder_paths = halve_csv_files(values_paths, halve_seed)

    # 3 — Build template CSV (one row per speaker, conditions all NaN)
    speaker_ids: dict[str, str] = {}
    for folder_path in wav_folder_paths:
        for wav_file in sorted(Path(folder_path).rglob("*.wav")):
            spk = _speaker_id_from_stem(wav_file.stem)
            if spk not in speaker_ids:
                speaker_ids[spk] = _parse_depression_diagnosis(spk)

    template_rows = [
        {
            "Utterance_File":       spk_id,
            "depression_diagnosis": diag,
            "F0_Mean":     np.nan,
            "F0_StdDev":   np.nan,
            "F0_Variance": np.nan,
            "toolbox":     np.nan,
            "noise":       np.nan,
            "compression": np.nan,
            "dataset_size": np.nan,
        }
        for spk_id, diag in sorted(speaker_ids.items())
    ]
    pd.DataFrame(template_rows).to_csv(output_csv, index=False)

    # 4 — Compression condition rows (mp3 vs wav)
    add_f0_stats_compression_to_csv(output_csv, [mp3_output_dir], f0_wav_folder_paths)

    # 5 — Halving condition rows (halved vs full)
    add_halved_f0_to_csv(output_csv, halved_folder_paths, f0_full_folder_paths)

    # 6 — Noise condition rows (noisy vs denoised)
    add_noisy_denoised_f0_to_csv(output_csv, noisy_folder_paths, denoised_folder_paths)

    # 7 — Toolbox condition rows (one row per tool per utterance)
    add_toolbox_f0_to_csv(output_csv, **toolbox_folder_paths)

    # 8 — Drop template rows and reshape to the README nine-column schema
    _CONDITION_COLS = ["toolbox", "noise", "compression", "dataset_size"]
    df = pd.read_csv(output_csv)
    df = df.dropna(subset=_CONDITION_COLS, how="all")
    df = df.rename(columns={
        "Utterance_File": "utterance_id",
        "F0_Mean":        "F0_mean",
        "F0_StdDev":      "F0_sd",
        "F0_Variance":    "F0_var",
    })[["utterance_id", "F0_mean", "F0_sd", "F0_var",
        "depression_diagnosis", "toolbox", "noise", "compression", "dataset_size"]]

    df.to_csv(output_csv, index=False)
    return output_csv