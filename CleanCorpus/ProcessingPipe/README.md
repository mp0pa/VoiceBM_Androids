# ProcessingPipe - Audacity Audio Cleaning Pipeline

Batch audio denoising and enhancement pipeline that processes WAV files through a combination of Python (noise reduction) and Audacity via its scripting interface (`mod-script-pipe`).

The noise profile required for noise reduction is captured **automatically** for each file: the pipeline uses [Respiro-EN](https://github.com/keums/respiro-en) to detect breath / silence regions in the audio, selects the best one, and feeds it to `noisereduce` as the noise sample — no manual profile capture needed.

## Prerequisites

- **Audacity** with the `mod-script-pipe` module enabled
- **Python 3** (3.8+ recommended)
- **Respiro-EN** installed at the path configured in `denoisepipe.py` (see Setup)
- Python packages: `torch`, `noisereduce`, `soundfile`, `numpy`, `intervaltree`
- Input audio files in `.wav` format

### Enabling mod-script-pipe in Audacity

1. Open Audacity
2. Go to **Edit > Preferences > Modules** (on Linux/Windows) or **Audacity > Preferences > Modules** (on macOS)
3. Set `mod-script-pipe` to **Enabled**
4. Restart Audacity

## Project Structure

```
ProcessingPipe/
├── denoisepipe.py          # Main script: batch denoising and Audacity effects
├── denoisemetrics.py       # Metrics script: PESQ / SI-SAR / STOI evaluation
├── AudacityMacros/
│   ├── Denoiseok.txt       # Reference macro (documents the processing chain; not used at runtime)
│   ├── GetProfile.txt      # Reference macro (documents the noise profile step; not used at runtime)
│   └── Join.txt            # Track joining macro (utility, run manually in Audacity)
└── README.md
```

Respiro-EN lives outside this repository (see Setup).

## How It Works

The pipeline communicates with a running Audacity instance through named pipes. For each WAV file in the input directory, it:

1. Runs **Respiro-EN** to detect breath/silence intervals (non-speech regions that carry background noise)
2. Picks the **longest qualifying silence** as the noise sample source
3. Applies **single-pass noise reduction in Python** (`noisereduce`) using the detected silence as the noise profile
4. Writes the denoised audio to a temporary WAV file
5. Imports the temp file into Audacity and applies the remaining effects via the scripting pipe: stereo-to-mono, high-pass filter, noise gate, EQ curve, normalization
6. Exports the processed audio as mono WAV to the output directory
7. Removes the track from Audacity and deletes the temp file

### Why noise reduction is done in Python, not in Audacity

Audacity 3.x's `mod-script-pipe` does not expose `GetNoiseProfile:` as a callable command, and macro execution (`Macro_*`, `ApplyMacro:`) is also unavailable through the pipe in this version. All other effects (`High-passFilter`, `NoiseGate`, `FilterCurve`, `Normalize`, `StereoToMono`) are directly available as pipe commands and run in Audacity as before.

## Setup

### 1. Install Respiro-EN

Clone or download [Respiro-EN](https://github.com/keums/respiro-en) and place the `respiro-en.pt` model weights inside it.  Then open `denoisepipe.py` and update the `RESPIRO_PATH` constant to point to that directory:

```python
RESPIRO_PATH = "/path/to/Respiro-en"
```

### 2. Install Python packages

```bash
pip install torch noisereduce soundfile numpy intervaltree
```

### 3. Configure Input/Output Directories

Open `denoisepipe.py` and update the two directory constants:

```python
in_dir  = "/path/to/input/audio"
out_dir = "/path/to/output/audio"
```

- `in_dir` — directory containing the `.wav` files to process
- `out_dir` — directory where cleaned files will be saved (must already exist)

## Running the Pipeline

1. **Start Audacity** (it must be running before the script)
2. Run the script:

```bash
python denoisepipe.py
```

The script logs each file, the silence segment used for noise profiling, and each Audacity command. All output files keep their original filenames and are written to `out_dir` as mono WAV.

## Processing Chain Reference

The pipeline applies the following steps in order:

| Step | Where | Effect | Key Parameters | Purpose |
|------|--------|--------|---------------|---------|
| 1 | Python | Noise Reduction | `noisereduce` defaults | Reduce background noise using the silence profile |
| 2 | Audacity | StereoToMono | — | Convert to single channel |
| 3 | Audacity | High-Pass Filter | Frequency=50 Hz, Rolloff=12 dB/oct | Remove low-frequency rumble |
| 4 | Audacity | Noise Gate | Threshold=-40 dB, Level-Reduction=-24 dB, Attack=10 ms, Decay=100 ms, Hold=50 ms | Gate out residual low-level noise in silent sections |
| 5 | Audacity | Filter Curve (EQ) | 17-point B-spline curve, 67 Hz – 20.8 kHz | Shape frequency response for speech clarity |
| 6 | Audacity | Normalize | Peak Level=-1 dB, Remove DC Offset=yes | Normalize volume, prevent clipping |

### Automatic noise profile capture (Respiro-EN)

For each file the pipeline calls `find_noise_profile_segment()`, which:

1. Runs `BreathDetector` (Respiro-EN) on the raw WAV file at a threshold of `0.064` and a minimum segment length of 20 frames (200 ms)
2. Picks the **longest detected interval** that is at least `MIN_SILENCE_DURATION` (default 0.3 s)
3. Returns `(start, end)` in seconds

The identified segment is passed to `denoise_audio()`, which extracts the corresponding samples as the noise profile for `noisereduce`.

**Fallback behaviour:** If Respiro-EN finds no silence that meets the minimum duration (or if detection fails), the pipeline falls back to the first 0.5 s of the file, which typically contains room tone before the speaker begins.

### AudacityMacros/ folder

The `.txt` files in `AudacityMacros/` are kept as **documentation and manual-use reference** only — they are not loaded or executed by the pipeline at runtime.

- `Denoiseok.txt` documents the full processing chain as an Audacity macro. It can be installed manually in Audacity (`~/.config/audacity/Macros/` on Linux) if you want to run the chain interactively.
- `GetProfile.txt` is the single-step `GetNoiseProfile:` macro, useful for manual experimentation.
- `Join.txt` aligns multiple tracks end-to-end and mixes them into one. Run it manually in Audacity for concatenation tasks.

## Customizing Parameters

### Noise-profile detection settings

In `denoisepipe.py`:

| Constant / argument | Default | What it controls |
|---------------------|---------|-----------------|
| `RESPIRO_PATH` | (path to Respiro-en) | Location of the Respiro-EN directory |
| `MIN_SILENCE_DURATION` | `0.3` s | Shortest interval accepted as a noise sample; increase for a stricter selection |
| `threshold` in `find_noise_profile_segment` | `0.064` | Respiro-EN detection threshold — lower values detect more (possibly shorter) breath regions |
| `min_length` in `find_noise_profile_segment` | `20` frames (200 ms) | Minimum frame run that counts as a detected breath interval |

### Noise reduction strength

`noisereduce` is called with default parameters. To adjust aggressiveness, pass extra arguments to `nr.reduce_noise()` in `denoise_audio()`:

```python
# More aggressive reduction
reduced = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample, prop_decrease=1.0, stationary=False)
```

Key parameters:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `prop_decrease` | `1.0` | Proportion of noise to reduce (0–1). Lower for gentler reduction. |
| `stationary` | `False` | `True` assumes noise is constant; `False` adapts over time |
| `n_std_thresh_stationary` | `1.5` | Sensitivity threshold for stationary mode |

### Audacity effect parameters

The Audacity pipe commands and their parameters are set directly in `apply_pipeline()` in `denoisepipe.py`. Key values:

| Parameter | Location in code | Default | What it controls |
|-----------|-----------------|---------|-----------------|
| High-pass frequency | `High-passFilter:` call | `90` Hz | Cutoff for low-frequency removal |
| High-pass rolloff | `High-passFilter:` call | `dB12` | Filter steepness. Options: `dB6`, `dB12`, `dB24`, `dB36`, `dB48` |
| Noise gate threshold | `NoiseGate:` call | `-40` dB | Level below which the gate closes |
| Noise gate level reduction | `NoiseGate:` call | `-24` dB | Attenuation when the gate is closed |
| Noise gate attack | `NoiseGate:` call | `10` ms | How fast the gate opens |
| Noise gate decay | `NoiseGate:` call | `100` ms | How fast the gate closes |
| Noise gate hold | `NoiseGate:` call | `50` ms | Minimum open time |
| EQ curve points | `FilterCurve:` call | (17 points, 67 Hz–20.8 kHz) | `f0`–`f16` = frequencies in Hz, `v0`–`v16` = gain in dB |
| Normalization peak | `Normalize:` call | `-1` dB | Target peak amplitude |

### Changing Output Channels

The export command forces mono output:

```python
do_command(f'Export2: Filename="{output_path}" NumChannels=1')
```

Change `NumChannels=1` to `NumChannels=2` for stereo output.

### Adjusting the Sleep Timer

A 0.1 s pause between files lets Audacity settle:

```python
time.sleep(0.1)
```

Increase this if Audacity struggles with rapid successive imports (e.g., large files or slow machines).

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `..does not exist. Ensure Audacity is running with mod-script-pipe.` | Start Audacity first and verify mod-script-pipe is enabled in Preferences > Modules |
| Audacity slows down over many files | The script already disables undo history during processing. If still slow, increase the sleep timer or process in smaller batches. |
| Noise reduction has no effect | Check the log: if every file fell back to the 0.0–0.5 s segment, Respiro-EN may not be detecting any silence. Try lowering `threshold` or `MIN_SILENCE_DURATION`. |
| `ModuleNotFoundError: No module named 'modules'` | `RESPIRO_PATH` in `denoisepipe.py` does not point to the correct Respiro-EN directory. |
| `ModuleNotFoundError: No module named 'noisereduce'` | Run `pip install noisereduce soundfile`. |
| An Audacity effect command fails | The command ID may differ across Audacity versions or locales. Run `python3 -c "..."` with a `GetInfo: Type=Commands` query to list all available IDs on your system. |

---

## Metrics Evaluation (`denoisemetrics.py`)

`denoisemetrics.py` compares every processed file in `out_dir` against its original counterpart in `in_dir` and writes the results to a CSV file.

### What it computes

| Metric | Full name | Range | Interpretation |
|--------|-----------|-------|----------------|
| **PESQ** | Perceptual Evaluation of Speech Quality | ≈ −0.5 – 4.5 MOS-LQO | Higher = better perceptual quality. Computed in wideband mode at 16 kHz. |
| **SI-SAR** | Scale-Invariant Signal-to-Artifacts Ratio | dB, higher is better | Measures how much processing artifact was introduced. Derived from the BSS eval framework; scale-invariant projection removes gain differences between reference and estimate. In single-source denoising there is no interference component, so the full error is artifacts. |
| **STOI** | Short-Time Objective Intelligibility | 0 – 1 | Higher = more intelligible speech. Uses the `pystoi` implementation (resampled to 10 kHz internally). |

The **reference** is the original file from `in_dir`; the **estimate** is the processed file from `out_dir`. This evaluates how the pipeline changes perceptual quality and intelligibility relative to the untouched recording (useful when a ground-truth clean recording is not available).

### Output CSV

A new file is created in `out_dir` on every run, named:

```
<output_folder_name>_<YYYYMMDD>_<HHMMSS>.csv
```

Example: `HC_cleaned_20260316_143022.csv`

Each row corresponds to one WAV file:

```
file,PESQ,SI-SAR,STOI
HC_001.wav,3.1274,18.4532,0.8761
HC_002.wav,2.9853,17.0214,0.8510
...
```

If a metric cannot be computed for a file (e.g., missing reference or mismatched format), the row contains `error` for that column and processing continues.

### Setup

Install the additional dependencies:

```bash
pip install pesq pystoi scipy
```

`soundfile` and `numpy` are already required by `denoisepipe.py`.

### Running

```bash
python denoisemetrics.py
```

Audacity does **not** need to be running. The script operates entirely in Python on the files already saved to disk.

### Implementation notes

#### Parallel processing

Files are processed concurrently using `ProcessPoolExecutor`, which spawns one worker process per CPU core. Each file is fully independent (read → resample → compute → write row), so all cores stay busy. Results are written to the CSV incrementally as each worker finishes, so the file is valid even if the script is interrupted mid-run.

#### Resampling

Resampling uses `scipy.signal.resample_poly` (polyphase FIR filter) rather than the FFT-based `scipy.signal.resample`. The two sample rates are reduced by their GCD first so the filter operates at the smallest integer up/down ratio — significantly faster for common pairs like 44 100 → 16 000 Hz.

#### PESQ chunking

The `pesq` C extension (ITU-T P.862 reference implementation) allocates fixed-size stack buffers sized for short utterances and crashes with a stack-smashing error on audio longer than ~10 s. To handle full interview recordings, each signal is split into 8-second non-overlapping chunks, PESQ is scored on each chunk, and the mean is returned. Chunks shorter than 0.5 s at the end of the file are discarded.

#### SI-SAR computation

The scale-invariant projection is computed without allocating intermediate arrays, using the identities:

```
‖s_target‖²  =  dot(est, ref)²  /  ‖ref‖²
‖e_artif‖²   =  ‖est‖²  −  ‖s_target‖²
```

Both signals are mean-centred before the dot products to make the result invariant to DC offset.