# Audio Segmentation and Feature Extraction Pipeline

This directory contains a pipeline to segment speech audio files into individual utterances, extract the fundamental frequency (F0) from each utterance and calculate its mean, standard deviation and variance.

The pipeline works in three main stages:
1.  **Timestamp Detection**: It uses the Respiro-EN model to detect non-speech breath intervals in the source audio files
2.  **Segmentation**: It uses the detected breath timestamps to slice the audio, isolating continuous speech segments (utterances)
3.  **F0 Extraction**: It processes each utterance using three different tools (openSMILE, librosa and parlselmouth) to extract the fundamental frequency (F0) and calculates mean, standard deviation and variance

## Prerequisites

- **Python 3** (3.8+ recommended)

- **[Respiro-en](https://github.com/ydqmkkx/Respiro-en)** downloaded locally

- **Python packages**: `torch`, `torchaudio`, `librosa`, `numpy`, `sox`, `opensmile`, `pandas`, `argparse`, `parselmouth`

- **[Sound eXchange](https://sourceforge.net/projects/sox/)** command-line utility must be installed on the OS:
    - On Debian/Ubuntu: `sudo apt install sox`
    - On macOS (using Homebrew): `brew install sox`
    - Or via the [download link](https://sourceforge.net/projects/sox/files/latest/download)

- Input audio files in `.wav` format

*Note: On Ubuntu/Pop!_OS, using Python 3.13 and newer SciPy builds, like the Respiro-en model, might require updating your C++ standard library (`sudo apt install --only-upgrade libstdc++6`).*

*System Note: Ensure the `torch` version is compatible with the system and CUDA version if using a GPU*


## Pipeline Structure

```text
feature_extraction/
├── F0_extraction/
│   ├── f0_extr_egemaps.py       # Script for F0 extraction with eGeMAPS
│   ├── f0_extr_praat.py         # Script for F0 extraction with Praat
│   └── f0_extr_yin.py           # Script for F0 extraction with pYIN
|── AndroidsResults/
│   ├── ...                      # See last paragraph for detailed structure
├── segmentation/
│   ├── merge_clips.py           # Utility to merge clips into a single file
│   ├── respiro_timestamps.py    # Script for timestamp generation
│   └── segment.py               # Script for audio segmentation
├── README.md
└── run_pipeline.sh              # Master script to execute the full pipeline
```

## Preliminary steps

### Install Python Packages

```bash
pip install torch torchaudio librosa numpy sox opensmile pandas argparse praat-parselmouth
```
Make sure that the versions of `torch` and `torchaudio` match

### (Optional) Merge audio clips

In case there are audio files divided into multiple clips, merge them together leveraging the `merge_clips.py` script

```bash
python feature_extraction/merge_clips.py path/to/input_dir path/to/<output_dir>
```

# Implementation (detailed)

## 1. Timestamp detection

The script `respiro_timestamps.py` automatically traverses both flat and nested directory structures to process audio files

For each WAV file, it:
1. Loads the audio and extracts features (Mel-spectrogram, zero-crossing rate, variance of Mel-spectrogram)
2. Runs the Respiro-en model to detect non-speech breath intervals
3. Filters the detected intervals to ensure they meet a minimum length threshold of 20 milliseconds
4. Calculates accurate `start_time_sec` and `end_time_sec` markers of breath intervals
5. Outputs this data into `.csv` files

Timestamps are outputted to `.csv` files so to be easily reviewed, manually corrected if necessary, and sequentially fed into external tools like `SoX` (Sound eXchange) to physically slice the WAV files into segmented clips without altering the original recordings during this step

### 

### Running the Pipeline

**I**. Install Respiro-en and the model weights
- Clone [Respiro-en](https://github.com/ydqmkkx/Respiro-en) locally.
- Download the pre-trained model weights (`respiro-en.pt`) from the repository and place them inside the cloned repository
- Open the script `feature_extraction/segmentation/respiro_timestamps.py` and update the `RESPIRO_PATH` variable to point to that directory:
```python
RESPIRO_PATH = "/path/to/Respiro-en"
```

You must manually download the `respiro-en.pt` weights file from the official Respiro-en repository and place it inside the `feature_extraction/segmentation/Respiro-en/` subdirectory before running the script

**II**. Execute the script from the root of the project repository 

Raw, non-denoised recordings, inputted as WAVE files, are expected to be available locally and are added as command line arguments

```bash
python feature_extraction/segmentation/respiro_timestamps.py path/to/raw_audio_directory path/to/<output_csv>
```

## 2. Segmentation

 `segment.py`, the second script in the audio processing pipeline, takes breath timestamps in CSV files and uses them to slice the corresponding raw WAVE files into smaller clips, each containing a single speech utterance

 The script:

 - Reads the input CSV file, which contains the file paths and the start/end times of detected _breath_ intervals
 - Organizes all the breath intervals from the CSV, grouping them by the original audio file they belong to 
 - For each audio file, determines the actual speech segments by "inverting" the breath intervals. Considering that a speech utterance is the audio that occurs _between_ the breath pauses. It also correctly identifies the speech from the very beginning of the file to the first breath, and from the end of the last breath to the end of the file
 - Creates a dedicated sub-folder for each original audio file to keep the segmented utterances organized. For example, when processing `speaker_1.wav`, a folder named `speaker_1_segmented/` will be created to hold the output clips
 - Using the `sox` library, trims the original audio file based on the start and end times of the calculated speech segments
 - Saves each trimmed speech segment as a new, sequentially numbered WAVE file (e.g., `speaker_1_utt001.wav`, ...,`speaker_N_utt002.wav`, etc.) inside its corresponding `speaker_1_segmented/` folder
 - Segments shorter than 0.2 seconds long are discarded to prevent files with spurious noise between breaths

 ### Running the pipeline

Execute the script from the root of the project repository

```bash
python feature_extraction/segmentation/segment.py path/to/raw_audio_directory path/to/<corresponding_timestamps.csv> path/to/save/<segmented_audio_directory>
```


## F0 extraction

 All the scripts for F0 extraction:
 - Isolate voiced speech from unvoiced frames so that mean, variance and standard deviation calculations are based on actual voiced speech
- For every single utterance file, create a CSV file inside the `<speaker>/values` directory containing the start time, end time, and F0 value (in Hz) for every 10ms frame
- For each speaker, aggregate the statistical data (mean, standard deviation, and variance) for all their utterances
- Save grouped statistics into three CSV files `<speaker>_mean.csv`, `<speaker>_standard_dev.csv`, `<speaker>_variance.csv`) inside their respective subdirectories within an over-arching `F0_results` folder

### Option I: eGeMAPS with (extended Geneva Minimalistic Acoustic Parameter Set) [`openSMILE`](https://audeering.github.io/opensmile-python/) 

The `f0_extr_egemaps.py` script:

- Initializes openSMILE to extract Low-Level Descriptor (LLDs), such as F0 value, which is given by default for every 10 millisecond frame of audio
- Converts semitones to Hz, since openSMILE outputs F0 in semitones (relative to 27.5 Hz), which are converted mathematically into Hz for easier interpretation
```python
f0_hz_series = f0_semitones.apply(lambda x: 27.5 * (2 ** (x / 12)) if x != 0 else 0.0)
```

### Option II: Probabilistic YIN with [`librosa`](https://librosa.org/doc/main/generated/librosa.pyin.html)

The script `f0_extr_librosa.py`:

- Uses librosa's probabilistic YIN (`pyin`) algorithm to track pitch, aligned to a 10 ms hop length.
- librosa represents unvoiced frames as `NaN`. The script converts these `NaN` values to `0.0` to ensure consistency across the other pipeline outputs

### Option III: Praat with [`parselmouth`](https://parselmouth.readthedocs.io/en/stable/api_reference.html#parselmouth.Pitch)

The script  `f0_extr_praat.py`:

- Uses the Parselmouth wrapper to access Praat's native pitch tracking (`to_pitch`), configured for a 10 ms time step
- Praat centers its frame timestamps (e.g., 0.005s). The script shifts these back to standard left-aligned start times (0.000s) to perfectly match the time series of the other tools. Unvoiced frames are naturally represented as `0.0`


### Running the script

```bash
python /path/to/f0_extr_<choose a tool>.py path/to/<segmented_audio_directory> path/to/<output_directory>
```
`<segmented_audio_directory>` is expected to be a nested directory where each subdirectory corresponds to a speaker

# Implementation (simplified)

A master shell script (`run_pipeline.sh`) is provided to execute the entire pipeline (timestamp detection, segmentation, and F0 extraction) with a single command

```bash
bash feature_extraction/run_pipeline.sh --raw-audio path/to/<raw_audio_dir> --output-dir path/to/<results_base_dir> --f0-method egemaps|yin|praat
```

**Arguments:**
- `--input`: The directory containing of `.wav` files, with the possibility of them being in nested subfolders.
- `--output`: The directory where the F0 CSV files will be saved.
- `--f0-method`: The extraction tool to use. Options are `egemaps`, `yin`, or `praat` (default: `praat`).

⚠️ `merge_clips.py` is not included here

# `AndroidsResults` directory organization

Based on the execution of the full pipeline, the structure of the results directory for [The Androids Corpus](https://github.com/androidscorpus/data) is organized as follows:

```text
AndroidsResults/
├── F0_results/                         # Extracted F0 results
│   ├── F0_egemaps/                     # Extracted F0 results using eGeMAPS (openSMILE)
│   │   ├── F0_results_IT/              # Results for Interview Task (example)
│   │   │   ├── ITspeaker_1_segmented/
│   │   │   │   ├── mean/                   # Mean F0 per utterance
│   │   │   │   │   └── ITspeaker_1_mean.csv
│   │   │   │   ├── standard_dev/           # Standard deviation of F0 per utterance
│   │   │   │   │   └── ITspeaker_1_standard_dev.csv
│   │   │   │   ├── values/                 # Frame-by-frame F0 values per utterance
│   │   │   │   │   ├── ITspeaker_1_utt001.csv
│   │   │   │   │   └── ...
│   │   │   │   └── variance/               # Variance of F0 per utterance
│   │   │   │       └── ITspeaker_1_variance.csv
│   │   │   └── ITspeaker_N_segmented/
│   │   │       └── ...
│   │   └── F0_results_RT/              # Results for other tasks...
│   │       └── ...
│   ├── F0_praat/                       # Extracted F0 results using Praat (parselmouth)
│   │   └── ...                         # (Same inner structure as above)
│   └── F0_yin/                         # Extracted F0 results using pYIN (librosa)
│       └── ...                         # (Same inner structure as above)
├── segmented/                          # Segmented utterances (from segment.py)
│   ├── IT_segmented/
│   │   ├── ITspeaker_1_segmented/
│   │   │   ├── speaker_1_utt001.wav
│   │   │   └── ...
│   │   └── ...
│   ├── ITok_segmented/
│   │   └── ...
│   ├── RT_segmented/
│   │   └── ...
│   └── RTok_segmented/
│       └── ...
└── timestamps/                         # Detected breath intervals (from respiro_timestamps.py)
    ├── timestamps_IT.csv
    ├── timestamps_ITok.csv
    ├── timestamps_RT.csv
    └── timestamps_RTok.csv

```
