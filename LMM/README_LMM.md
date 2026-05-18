# LMM preprocessing pipeline

In order to use LMM system to accomplish verification and analytical validation, this pipeline will prepare the data to be well organized to be use into a R algorithm to compute LMM analysis.

Verification will be the first step while verifying the integrity of the computing process part's accuracy seems the basis for the following work.

---

## verification_prep_data.py

### Purpose

Prepare a clean `.csv` file to be used in R for fitting a **Linear Mixed Model (LMM)** that studies the influence of other modalities on F0 (mean, standard deviation, and variance).

---

### Target output

The final `.csv` contains 9 columns:

| Column | Description |
|---|---|
| `utterance_id` | Unique identifier for each utterance, encoding the original filename + modality + level |
| `F0_mean` | Mean F0 (Hz, voiced frames only) |
| `F0_sd` | Standard deviation of F0 (Hz, voiced frames only) |
| `F0_var` | Variance of F0 (Hz, voiced frames only) |
| `depression_diagnosis` | Depression label for the speaker (`C` = control, `P` = patient) |
| `toolbox` | F0 extraction toolbox used |
| `noise` | Noise condition applied to the audio |
| `compression` | Audio compression applied |
| `dataset_size` | Size of the dataset used |

#### Utterance ID convention

Each `utterance_id` encodes the original filename and the condition observed:

```
<original_stem>_<modality>_<level>
```

Examples:
- `IT_01_CF56_1_utt001_compression_mp3`
- `IT_01_CF56_1_utt001_noise_denoised`
- `IT_01_CF56_1_utt001_halving_halved`
- `IT_01_CF56_1_utt001_toolbox_praat`

#### Utterance filename format

Original `.wav` filenames follow the pattern:

```
TASK_PATIENTID_DIAGSEXAGE_EDUCATION_UTT.wav
e.g.  IT_01_CF56_1_utt001.wav
      ^^  ^^  ^^ ^^  ^
      |   |   |  |   utterance index
      |   |   |  education level
      |   |   age (56)
      |   |   diagnosis + sex: C/P + F/M  (CF = Control Female)
      |   patient id
      task id (IT or RT)
```

`depression_diagnosis` is extracted from the first character of the diagnosis+sex+age field: `C` (control) or `P` (patient).

#### Base (reference) condition levels

When a row is created for a given condition, all other condition columns are filled with the reference level:

| Column | Base level |
|---|---|
| `toolbox` | `egemaps` |
| `noise` | `noisy` |
| `compression` | `wav` |
| `dataset_size` | `full` |

---

### Functions

#### Preprocessing utilities

##### `wav_to_mp3(folder_paths, output_dir, bitrate)`

Converts every `.wav` file found in a list of folders to `.mp3` and saves them to `output_dir`.

| Parameter | Type | Description |
|---|---|---|
| `folder_paths` | `list[str]` | Folders containing `.wav` files |
| `output_dir` | `str` | Destination folder for converted `.mp3` files (default: `"mp3_converted"`) |
| `bitrate` | `str` | MP3 bitrate, e.g. `"128k"` (default: `"128k"`) |

Returns: `list[str]` — paths to generated `.mp3` files.

---

##### `halve_csv_files(folder_paths, seed)`

For each folder, randomly removes half the data rows from every `.csv` file and writes the result to a sibling folder named `<folder>_halved`. Original files are never modified.

| Parameter | Type | Description |
|---|---|---|
| `folder_paths` | `list[str]` | Folders containing frame-level F0 `.csv` files to halve |
| `seed` | `int \| None` | Optional random seed for reproducibility |

Returns: `list[str]` — paths to the created `_halved` output folders.

---

#### Row-append pipeline functions

Each function reads the current state of the main CSV, then **appends one new row per utterance per condition level**. All condition columns (`toolbox`, `noise`, `compression`, `dataset_size`) are fully populated in every appended row using the base reference level for conditions not set by the current function.

---

##### `add_f0_stats_compression_to_csv(csv_path, mp3_folder_paths, f0_csv_folder_paths)`

Appends **2 rows per utterance** — one for the `mp3` compression level and one for the `wav` level.

- **mp3 row**: F0 extracted live from the `.mp3` file using `librosa.pyin`.
- **wav row**: F0 read from pre-computed stats in `f0_csv_folder_paths`.

Condition column set: `compression` → `"mp3"` or `"wav"`.

| Parameter | Type | Description |
|---|---|---|
| `csv_path` | `str` | Main CSV to enrich (saved in place) |
| `mp3_folder_paths` | `list[str]` | Folders containing `.mp3` files |
| `f0_csv_folder_paths` | `list[str]` | Speaker folders with `mean/`, `standard_dev/`, `variance/` sub-folders holding pre-computed per-utterance F0 stats from `.wav` files |

---

##### `add_halved_f0_to_csv(csv_path, halved_folder_paths, f0_csv_folder_paths)`

Appends **2 rows per utterance** — one for the `halved` dataset size and one for the `full` level.

- **halved row**: F0 computed from frame-level halved CSVs (voiced frames where `F0_Hz > 0`).
- **full row**: F0 read from pre-computed stats in `f0_csv_folder_paths`.

Condition column set: `dataset_size` → `"halved"` or `"full"`.

| Parameter | Type | Description |
|---|---|---|
| `csv_path` | `str` | Main CSV to enrich (saved in place) |
| `halved_folder_paths` | `list[str]` | Folders containing per-utterance halved frame-level F0 CSVs (`Start_Time`, `End_Time`, `F0_Hz`) |
| `f0_csv_folder_paths` | `list[str]` | Speaker folders with `mean/`, `standard_dev/`, `variance/` sub-folders for the full dataset |

---

##### `add_noisy_denoised_f0_to_csv(csv_path, noisy_folder_paths, denoised_folder_paths)`

Appends **2 rows per utterance** — one for the `noisy` condition and one for the `denoised` condition.

F0 stats are read from pre-computed stats in the respective folder paths.

Condition column set: `noise` → `"noisy"` or `"denoised"`.

| Parameter | Type | Description |
|---|---|---|
| `csv_path` | `str` | Main CSV to enrich (saved in place) |
| `noisy_folder_paths` | `list[str]` | Speaker folders (mean/standard_dev/variance) for the noisy condition |
| `denoised_folder_paths` | `list[str]` | Speaker folders (mean/standard_dev/variance) for the denoised condition |

---

##### `add_toolbox_f0_to_csv(csv_path, **toolbox_folder_paths)`

Appends **one row per utterance per toolbox**. Toolboxes are passed as keyword arguments.

Condition column set: `toolbox` → the tool name (e.g. `"egemaps"`, `"praat"`, `"librosa"`).

| Parameter | Type | Description |
|---|---|---|
| `csv_path` | `str` | Main CSV to enrich (saved in place) |
| `**toolbox_folder_paths` | `list[str]` per keyword | Each keyword is a tool name; its value is a list of speaker folders (mean/standard_dev/variance) for that tool's F0 output |

---

#### Entry point

##### `main(output_csv, wav_folder_paths, mp3_output_dir, f0_wav_folder_paths, f0_values_folder_paths, f0_full_folder_paths, noisy_folder_paths, denoised_folder_paths, toolbox_folder_paths, mp3_bitrate, halve_seed)`

Orchestrates the full pipeline and produces the final 9-column LMM CSV.

| Parameter | Type | Description |
|---|---|---|
| `output_csv` | `str` | Path to the output `.csv` file |
| `wav_folder_paths` | `list[str]` | Folders with original `.wav` utterances (utterance list source) |
| `mp3_output_dir` | `str` | Destination for converted `.mp3` files |
| `f0_wav_folder_paths` | `list[str]` | Speaker folders (mean/standard_dev/variance) for `.wav` F0 stats — wav side of compression condition |
| `f0_values_folder_paths` | `list[str]` | Folders with frame-level F0 value CSVs to halve |
| `f0_full_folder_paths` | `list[str]` | Speaker folders (mean/standard_dev/variance) for full-dataset F0 stats — full side of halving condition |
| `noisy_folder_paths` | `list[str]` | Speaker folders (mean/standard_dev/variance) for noisy F0 stats |
| `denoised_folder_paths` | `list[str]` | Speaker folders (mean/standard_dev/variance) for denoised F0 stats |
| `toolbox_folder_paths` | `dict[str, list[str]]` | Mapping `tool_name → speaker folders` for each toolbox |
| `mp3_bitrate` | `str` | MP3 bitrate (default `"128k"`) |
| `halve_seed` | `int \| None` | Random seed for halving reproducibility |

**Pipeline steps:**

1. Convert `.wav` → `.mp3` (`wav_to_mp3`)
2. Halve frame-level F0 CSVs (`halve_csv_files`) — returns `halved_folder_paths`
3. Build template CSV (one row per utterance with `utterance_id` + `depression_diagnosis`, all conditions `NaN`)
4. Append compression rows (`add_f0_stats_compression_to_csv`)
5. Append halving rows (`add_halved_f0_to_csv`)
6. Append noise rows (`add_noisy_denoised_f0_to_csv`)
7. Append toolbox rows (`add_toolbox_f0_to_csv`)
8. Drop template rows, rename columns to the 9-column schema, save

---

### How to run the pipeline

```python
from verification_prep_data import main

main(
    output_csv="lmm_verification.csv",

    # Original .wav utterances (used as utterance list and for wav→mp3 conversion)
    wav_folder_paths=[
        "feature_extraction/AndroidsResults/segmented/IT_01_CF56_1_segmented",
        "feature_extraction/AndroidsResults/segmented/IT_02_CM57_2_segmented",
    ],

    # Where to store converted .mp3 files
    mp3_output_dir="LMM/mp3_converted",

    # Pre-computed F0 stats from original .wav (wav side of compression condition)
    # Each entry is a speaker folder containing mean/, standard_dev/, variance/
    f0_wav_folder_paths=[
        "feature_extraction/AndroidsResults/F0_results/F0_yin/F0_results_IT/IT_01_CF56_1_segmented",
        "feature_extraction/AndroidsResults/F0_results/F0_yin/F0_results_IT/IT_02_CM57_2_segmented",
    ],

    # Frame-level F0 value CSVs to halve (values/ folders)
    f0_values_folder_paths=[
        "feature_extraction/AndroidsResults/F0_results/F0_yin/F0_results_IT/IT_01_CF56_1_segmented/values",
        "feature_extraction/AndroidsResults/F0_results/F0_yin/F0_results_IT/IT_02_CM57_2_segmented/values",
    ],

    # Pre-computed F0 stats from full dataset (full side of halving condition)
    f0_full_folder_paths=[
        "feature_extraction/AndroidsResults/F0_results/F0_yin/F0_results_IT/IT_01_CF56_1_segmented",
        "feature_extraction/AndroidsResults/F0_results/F0_yin/F0_results_IT/IT_02_CM57_2_segmented",
    ],

    # Pre-computed F0 stats for noisy and denoised conditions
    noisy_folder_paths=[
        "feature_extraction/AndroidsResults/F0_results/F0_yin/F0_results_IT_noisy",
    ],
    denoised_folder_paths=[
        "feature_extraction/AndroidsResults/F0_results/F0_yin/F0_results_IT_denoised",
    ],

    # F0 extraction toolboxes: tool name → speaker folders (mean/standard_dev/variance)
    toolbox_folder_paths={
        "egemaps": [
            "feature_extraction/AndroidsResults/F0_results/F0_egemaps/F0_results_IT/IT_01_CF56_1_segmented",
        ],
        "praat": [
            "feature_extraction/AndroidsResults/F0_results/F0_praat/F0_results_IT/IT_01_CF56_1_segmented",
        ],
        "librosa": [
            "feature_extraction/AndroidsResults/F0_results/F0_librosa/F0_results_IT/IT_01_CF56_1_segmented",
        ],
    },

    # Optional
    mp3_bitrate="128k",
    halve_seed=42,
)
```

The output file `lmm_verification.csv` is ready to be loaded directly into R for LMM fitting.

#### Expected folder structure for pre-computed F0 stats

Each speaker folder passed to the pipeline functions must follow this layout:

```
IT_01_CF56_1_segmented/
├── mean/
│   └── IT_01_CF56_1_segmented_mean.csv          # columns: Utterance_File, F0_Mean
├── standard_dev/
│   └── IT_01_CF56_1_segmented_standard_dev.csv  # columns: Utterance_File, F0_StdDev
└── variance/
    └── IT_01_CF56_1_segmented_variance.csv       # columns: Utterance_File, F0_Variance
```

Frame-level value folders (used for halving) must contain per-utterance CSVs:

```
values/
├── IT_01_CF56_1_utt001.csv   # columns: Start_Time, End_Time, F0_Hz
├── IT_01_CF56_1_utt002.csv
└── ...
```