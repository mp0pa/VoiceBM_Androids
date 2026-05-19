from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf

_F0_DTYPE = {"F0_Mean": "float64", "F0_StdDev": "float64", "F0_Variance": "float64"}

_AGE_THRESHOLD = 47  # boundary used to bin speakers into age_group (<47 vs >=47)


def _build_f0_index(folder_paths: list[str]) -> dict[str, tuple]:
    """
    Walk a list of F0 extraction output folders and build a speaker_id →
    (mean, std, var) lookup, aggregating all utterances that belong to the
    same speaker.

    Expected structure (output of extract_f0 scripts):
        <folder>/
          <speaker_dir>/
            mean/        <speaker>_mean.csv        → Utterance_File, F0_Mean
            standard_dev/<speaker>_standard_dev.csv → Utterance_File, F0_StdDev
            variance/    <speaker>_variance.csv     → Utterance_File, F0_Variance

    Aggregation: F0_Mean = mean of utterance means;
                 F0_Variance = mean of utterance variances + variance of utterance means
                 (law of total variance); F0_StdDev = sqrt(F0_Variance).

    Args:
        folder_paths: Mother folders each containing speaker sub-folders with
                      mean/, standard_dev/, and variance/ sub-folders.

    Returns:
        Dict mapping speaker_id → (agg_mean, agg_std, agg_var).
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


def _build_metadata_index(metadata_csv_path: str) -> dict[str, str]:
    """
    Parse the metadata CSV and return a full_speaker_id → sex mapping.

    The metadata CSV has two side-by-side tables:
      - Columns 0–6 for RT speakers  (col 0 = speaker_id, col 3 = SEX)
      - Columns 7–13 for IT speakers (col 7 = speaker_id, col 10 = SEX)

    Speaker IDs in the CSV are enclosed in single quotes and may carry a
    trailing 'ok' suffix (e.g. '01_CF56_1ok').  Both are stripped so that
    the returned keys match the format produced by _speaker_id_from_stem
    (e.g. "IT_01_CF56_1", "RT_01_CF56_1").

    Args:
        metadata_csv_path: Path to the metadata .csv file.

    Returns:
        Dict mapping full speaker IDs to their sex label ("F" or "M").
    """
    df = pd.read_csv(metadata_csv_path, header=0)
    index: dict[str, str] = {}

    for _, row in df.iterrows():
        # RT table: columns 0–6
        raw_rt = str(row.iloc[0]).strip()
        if raw_rt not in ("nan", ""):
            spk_id = raw_rt.strip("'").removesuffix("ok")
            sex = str(row.iloc[3]).strip()
            if sex in ("F", "M"):
                index[f"RT_{spk_id}"] = sex

        # IT table: columns 7–13
        raw_it = str(row.iloc[7]).strip()
        if raw_it not in ("nan", ""):
            spk_id = raw_it.strip("'").removesuffix("ok")
            sex = str(row.iloc[10]).strip()
            if sex in ("F", "M"):
                index[f"IT_{spk_id}"] = sex

    return index


def _build_duration_index(audio_clipok_path: str) -> dict[str, float]:
    """
    Walk the InterviewTask audio clip folder and build a speaker_id →
    mean_speech_turn_duration (seconds) mapping.

    Speech turn duration is only available for the Interview Task (IT).

    Expected structure:
        <audio_clipok_path>/
          <PATIENTID_DIAGSEXAGE_EDUCATIONok>/      ← one folder per patient
            <PATIENTID_DIAGSEXAGE_EDUCATION_Xok>.wav  ← one file per turn

    Speaker ID is built by stripping the trailing 'ok' from the folder name
    and prepending 'IT_' (Interview Task prefix).

    Duration is read from the audio file header only (no decoding), using soundfile.
    The per-speaker value is the mean duration across all their speech turn files.

    Args:
        audio_clipok_path: Path to the audio_clipok directory
                           (CleanCorpus/InterviewTaskok/audio_clipok).

    Returns:
        Dict mapping speaker_id → mean speech turn duration in seconds.
    """
    src_dir = Path(audio_clipok_path)
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {src_dir}")

    speaker_durations: dict[str, list[float]] = {}

    for spk_dir in sorted(src_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk_id = "IT_" + spk_dir.name.removesuffix("ok")
        durations: list[float] = []
        for wav_file in sorted(spk_dir.glob("*.wav")):
            try:
                dur = sf.info(str(wav_file)).duration
                durations.append(float(dur))
            except Exception as e:
                print(f"  [Error] {wav_file.name}: {e}")
        if durations:
            speaker_durations[spk_id] = durations

    return {
        spk: float(np.mean(durs))
        for spk, durs in speaker_durations.items()
    }


def _speaker_id_from_stem(stem: str) -> str:
    """Strip the _utt### suffix (and optional trailing 'ok') to get the speaker identifier."""
    idx = stem.find("_utt")
    spk = stem[:idx] if idx != -1 else stem
    return spk.removesuffix("ok")


def _parse_depression_diagnosis(utterance_stem: str) -> str:
    """
    Extract the depression diagnosis label from a speaker ID stem.

    Expected format: TASK_PATIENTID_DIAGSEXAGE_EDUCATION
        e.g. IT_01_CF56_1 → first char of DIAGSEXAGE field → "C"

    Returns "C" (control), "P" (patient), or "unknown".
    """
    parts = utterance_stem.split("_")
    if len(parts) >= 3 and parts[2] and parts[2][0] in ("C", "P"):
        return parts[2][0]
    return "unknown"


def _parse_age_from_stem(stem: str) -> Optional[int]:
    """
    Extract the speaker's age from a speaker ID stem.

    Expected format: TASK_PATIENTID_DIAGSEXAGE_EDUCATION
        e.g. IT_01_CF56_1 → numeric suffix of DIAGSEXAGE → 56

    Returns the age as an integer, or None if the stem cannot be parsed.
    """
    parts = stem.split("_")
    if len(parts) >= 3 and len(parts[2]) > 2:
        age_str = parts[2][2:]      # strip two-char DIAG+SEX prefix (e.g. "CF" → "56")
        if age_str.isdigit():
            return int(age_str)
    return None


def _parse_task_from_stem(stem: str) -> str:
    """
    Extract the task label from a speaker ID stem.

    Expected format: TASK_PATIENTID_...
        e.g. IT_01_CF56_1 → "IT"

    Returns "IT", "RT", or "unknown".
    """
    task = stem.split("_")[0]
    return task if task in ("IT", "RT") else "unknown"


def main(
    output_csv: str,
    metadata_csv: str,
    f0_folder_paths: list[str],
    it_audio_clip_path: str,
) -> str:
    """
    Build the complete LMM analytical validation CSV.

    Analytical validation proves that no speaker-specific characteristic
    (sex, age, task, speech turn duration) can influence the capacity of F0
    (mean/SD/variance) to serve as a biomarker of depression.

    Each row in the output CSV represents one speaker with their aggregated F0
    statistics and all four analytical subgroup variables, ready to be used
    directly in R LMM models of the form:

        lme(F0_mean ~ depression_diagnosis, random = ~1 | sex)
        lme(F0_mean ~ depression_diagnosis, random = ~1 | age_group)
        lme(F0_mean ~ depression_diagnosis, random = ~1 | task)
        lme(F0_mean ~ depression_diagnosis, random = ~1 | speech_turn_duration)
        ...and their combinations.

    Output CSV schema (10 columns):
        utterance_id          — speaker identifier
        F0_mean               — aggregated mean F0 across utterances (Hz)
        F0_sd                 — aggregated F0 standard deviation (Hz)
        F0_var                — aggregated F0 variance (Hz²)
        depression_diagnosis  — "C" (control) or "P" (patient)
        sex                   — "F" or "M" (from metadata CSV)
        age                   — numeric age (parsed from filename)
        age_group             — "<47" or ">=47" (binned at _AGE_THRESHOLD)
        task                  — "IT" or "RT" (parsed from filename)
        speech_turn_duration  — mean speech turn duration in seconds (IT only; NaN for RT)

    Pipeline:
      1. Build the F0 index from pre-computed speaker stats in f0_folder_paths.
      2. Build the sex index from metadata_csv.
      3. Build the speech turn duration index from it_audio_clip_path.
      4. Discover all speakers from the F0 index.
      5. Assemble one row per speaker combining F0, diagnosis, and all subgroup
         variables; write the final CSV.

    Args:
        output_csv:         Path to the final output CSV file.
        metadata_csv:       Path to the metadata CSV used for sex lookup.
                            Expected two-table layout: RT speakers in columns 0–6,
                            IT speakers in columns 7–13 (col 3 / col 10 = SEX).
        f0_folder_paths:    Mother folders each containing speaker sub-folders that
                            hold mean/, standard_dev/, and variance/ sub-folders with
                            pre-computed per-utterance F0 stats (same format as used
                            by verification_prep_data.py).
        it_audio_clip_path: Path to CleanCorpus/InterviewTaskok/audio_clipok.
                            Each sub-folder is one IT patient; each .wav inside is
                            one speech turn. Used to compute mean speech turn duration
                            per IT speaker (RT speakers receive NaN).

    Returns:
        Path to the completed CSV file.
    """
    # 1 — F0 index: speaker_id → (mean, sd, var)
    f0_index = _build_f0_index(f0_folder_paths)

    # 2 — Sex index: speaker_id → "F" | "M"
    sex_index = _build_metadata_index(metadata_csv)

    # 3 — Duration index: IT speaker_id → mean speech turn duration (seconds)
    duration_index = _build_duration_index(it_audio_clip_path)

    # 4 — Discover all speakers from the F0 index
    speaker_ids = sorted(f0_index.keys())

    # 5 — Assemble one row per speaker
    rows = []
    for spk in speaker_ids:
        f0_mean, f0_sd, f0_var = f0_index.get(spk, (np.nan, np.nan, np.nan))
        age = _parse_age_from_stem(spk)
        age_group = (
            f"<{_AGE_THRESHOLD}" if age is not None and age < _AGE_THRESHOLD
            else f">={_AGE_THRESHOLD}" if age is not None
            else "unknown"
        )
        rows.append({
            "utterance_id":         spk,
            "F0_mean":              f0_mean,
            "F0_sd":                f0_sd,
            "F0_var":               f0_var,
            "depression_diagnosis": _parse_depression_diagnosis(spk),
            "sex":                  sex_index.get(spk, "unknown"),
            "age":                  age,
            "age_group":            age_group,
            "task":                 _parse_task_from_stem(spk),
            "speech_turn_duration": duration_index.get(spk, np.nan),
        })

        if spk not in sex_index:
            print(f"  [Warning] No sex metadata found for speaker '{spk}'.")
        if _parse_task_from_stem(spk) == "IT" and spk not in duration_index:
            print(f"  [Warning] No audio clips found for IT speaker '{spk}', duration set to NaN.")

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    return output_csv