# Audio Segmentation

This directory contains the tools necessary to analyze audio files from the VoiceBM Androids corpus and generate precise segmentation timestamps based on breath detection.

## Overview

The main script, `respiro_timestamps.py`, leverages the Respiro-en deep learning model. It automatically traverses both flat (Reading Task - RT) and nested (Interview Task - IT) directory structures to process audio files. 

For every WAV file, it detects non-speech breath intervals and calculates accurate `start_time_sec` and `end_time_sec` markers. This data is output into `.csv` files, which are intended to be fed into `SoX` (Sound eXchange) to physically slice the WAV files into segmented clips.

## Setup & Requirements

1. **Initialize the Submodule:**
   `Respiro-en` is included as a Git submodule. If the folder is empty after cloning, run:
   ```bash
   git submodule update --init --recursive
   ```

2. **Model Weights:**
   You must manually download the `respiro-en.pt` weights file from the official Respiro-en repository and place it inside the `Respiro-en/` subdirectory before running the script.

3. **System Dependencies:**
   The script requires a working installation of PyTorch, TorchAudio, and Librosa. 
   *Note: On Ubuntu/Pop!_OS, using Python 3.13 and newer SciPy builds might require updating your C++ standard library (`sudo apt install --only-upgrade libstdc++6`).*

## Usage

To ensure relative paths resolve correctly, the script should be executed from the root of the project repository:

```bash
python feature_extraction/segmentation/respiro_timestamps.py
```

## Outputs

The script safely ignores structural differences between directories and outputs the following files in your current working directory:

- `timestamps_IT.csv`
- `timestamps_RT.csv`
- `timestamps_ITok.csv`
- `timestamps_RTok.csv`