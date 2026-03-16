#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Audacity batch audio denoising pipeline with automatic noise profile capture.

For each WAV file the pipeline:
  1. Detects breath/silence regions with Respiro-EN (no speech, background noise only).
  2. Applies noise reduction (×1) in Python using the detected silence as the noise sample.
  3. Imports the denoised file into Audacity and applies the remaining effects via the
     scripting pipe: stereo-to-mono, high-pass filter, noise gate, EQ curve, normalization.
  4. Exports the processed file as mono WAV.

Requires Audacity to be running with mod-script-pipe enabled before launching.
"""

import os
import sys
import tempfile
import time

import noisereduce as nr
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Respiro-EN – automatic silence / breath detection
# ---------------------------------------------------------------------------
RESPIRO_PATH = "/home/mae/Documents/stage_L3_software/respiro_en/Respiro-en"
sys.path.insert(0, RESPIRO_PATH)

from modules import DetectionNet, BreathDetector  # noqa: E402

# Minimum duration (seconds) a detected silence must have to be used as the
# noise-profile source.  Intervals shorter than this are ignored; if none
# qualify the pipeline falls back to the first 0.5 s of the file.
MIN_SILENCE_DURATION = 0.3


def init_breath_detector():
    """Load the Respiro-EN model once for the whole session.

    Returns a ready-to-call BreathDetector instance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        os.path.join(RESPIRO_PATH, "respiro-en.pt"),
        map_location=device,
        weights_only=False,
    )
    model = DetectionNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Respiro-EN loaded on {device}.")
    return BreathDetector(model, device=device)


def find_noise_profile_segment(detector, wav_path, min_duration=MIN_SILENCE_DURATION):
    """Return (start, end) in seconds of the best silence for noise profiling.

    Uses Respiro-EN to find breath / pause intervals (non-speech regions that
    carry background noise).  The longest qualifying interval is chosen so that
    the noise reduction has the most representative noise sample.

    Falls back to the first 0.5 s of the file when:
      - Respiro-EN detects no intervals, or
      - all detected intervals are shorter than *min_duration*, or
      - an error occurs during detection.
    """
    try:
        tree = detector(wav_path, threshold=0.064, min_length=20)
        if tree:
            best = max(sorted(tree), key=lambda iv: iv.end - iv.begin)
            if (best.end - best.begin) >= min_duration:
                return best.begin, best.end
    except Exception as exc:
        print(f"    [Respiro-EN] Detection failed ({exc}); using fallback segment.")

    # Fallback: the very beginning of the file typically contains room tone
    # before the speaker starts talking.
    return 0.0, 0.5


# ---------------------------------------------------------------------------
# Python-based noise reduction
# ---------------------------------------------------------------------------

def denoise_audio(wav_path, noise_start, noise_end):
    """Apply single-pass noise reduction in Python and return the path to a temp WAV.

    Loads *wav_path*, extracts the silence segment [noise_start, noise_end] as
    the noise profile, and runs noisereduce once.  Writes the result to a temporary WAV file
    and returns its path.  The caller is responsible for deleting the temp file.

    noisereduce expects shape (channels, samples) for multichannel audio.
    soundfile reads shape (samples, channels), so we transpose before and after.
    """
    data, sr = sf.read(wav_path)

    # Transpose to (channels, samples) if stereo, keep 1-D if already mono.
    multichannel = data.ndim > 1
    if multichannel:
        audio = data.T  # (channels, samples)
        noise_sample = audio[:, int(noise_start * sr):int(noise_end * sr)]
    else:
        audio = data
        noise_sample = audio[int(noise_start * sr):int(noise_end * sr)]

    # Single denoising pass.
    reduced = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_sample)

    # Transpose back to (samples, channels) for soundfile.
    if multichannel:
        reduced = reduced.T

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, reduced, sr)
    return tmp.name


# ---------------------------------------------------------------------------
# Audacity pipe setup
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    TONAME = "\\\\.\\pipe\\ToSrvPipe"
    FROMNAME = "\\\\.\\pipe\\FromSrvPipe"
    EOL = "\r\n\0"
elif sys.platform == "darwin":
    TONAME = "/tmp/audacity_script_pipe.to." + str(os.getuid())
    FROMNAME = "/tmp/audacity_script_pipe.from." + str(os.getuid())
    EOL = "\n"
else:  # Linux
    TONAME = "/tmp/audacity_script_pipe.to." + str(os.getuid())
    FROMNAME = "/tmp/audacity_script_pipe.from." + str(os.getuid())
    EOL = "\n"

if not os.path.exists(TONAME):
    print(f"{TONAME} ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
    sys.exit()
if not os.path.exists(FROMNAME):
    print(f"{FROMNAME} ..does not exist.  Ensure Audacity is running with mod-script-pipe.")
    sys.exit()

TOFILE = open(TONAME, "w")
FROMFILE = open(FROMNAME, "rt")


def send_command(command):
    """Send a single command to Audacity."""
    TOFILE.write(command + EOL)
    TOFILE.flush()


def get_response():
    """Read and return Audacity's response to the last command."""
    result = ""
    line = ""
    while True:
        result += line
        line = FROMFILE.readline()
        if line == "\n" and len(result) > 0:
            break
    return result


def do_command(command):
    """Send one command and return the response."""
    send_command(command)
    response = get_response()
    print(f"[cmd] {command!r}  →  {response.strip()!r}")
    return response


# ---------------------------------------------------------------------------
# Directory configuration – update these to match your local paths
# ---------------------------------------------------------------------------
in_dir = "/home/mae/Documents/idmc/master1/university/s2/supervised_project/corpus/Androids-Corpus/Androids-Corpus/Interview-Task/audio/HC"
out_dir = "/home/mae/Documents/idmc/master1/university/s2/supervised_project/corpus/Androids-Corpus/Androids-Corpus/Interview-Task/audio/HC_cleaned"

filenames = [f for f in os.listdir(in_dir) if f.lower().endswith(".wav")]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def apply_pipeline(filenames, in_dir, out_dir, detector):
    """Process every WAV file through the full denoising pipeline.

    For each file:
      1. Find the best silence segment with Respiro-EN.
      2. Apply two-pass noise reduction in Python (noisereduce).
      3. Import the denoised temp file into Audacity.
      4. Apply the remaining effects via the scripting pipe:
           StereoToMono → High-pass filter → Noise gate → EQ curve → Normalize.
      5. Export as mono WAV and remove the track.
    """
    # Disable undo history to prevent Audacity from slowing down across files.
    do_command("SetPreference: Name=GUI/MaxUndoLevels Value=0 Reload=0")
    
    for f in filenames:
        input_path = os.path.abspath(os.path.join(in_dir, f))
        output_path = os.path.abspath(os.path.join(out_dir, f))

        print(f"\n--- Processing: {f} ---")

        # Step 1 – find noise segment.
        start, end = find_noise_profile_segment(detector, input_path)
        print(f"    Noise profile segment: {start:.3f}s – {end:.3f}s")

        # Step 2 – Python noise reduction; produces a temp file.
        print("    Running noise reduction (1 pass)...")
        denoised_path = denoise_audio(input_path, start, end)

        # Step 3 – import the denoised temp file into Audacity.
        do_command(f'Import2: Filename="{denoised_path}"')

        # Step 4 – apply remaining effects on the full track.
        do_command("SelectAll:")
        do_command("StereoToMono:")

        # Re-select after StereoToMono replaces the stereo track with a mono one.
        do_command("SelectAll:")
        do_command('High-passFilter: Frequency=50 RolloffType="dB12"')
        do_command(
            'NoiseGate: ATTACK="10" DECAY="100" GATE-FREQ="0" HOLD="50"'
            ' LEVEL-REDUCTION="-24" MODE="Gate" STEREO-LINK="LinkStereo" THRESHOLD="-40"'
        )
        do_command(
            'FilterCurve:'
            ' f0="67.516154" f1="94.215554" f2="101.80691" f3="120.73062" f4="963.29991"'
            ' f5="2011.3169" f6="3652.7602" f7="6633.7916" f8="10080.842" f9="13324.579"'
            ' f10="14398.197" f11="15558.322" f12="18593.811" f13="19328.392"'
            ' f14="20091.994" f15="20405.816" f16="20885.763"'
            ' v0="-18.42572" v1="-3.3924615" v2="-1.1308193" v3="-0.066518784"'
            ' v4="1.1308211" v5="1.1308211" v6="1.2638587" v7="0.066518784"'
            ' v8="-0.33259392" v9="-0.066518784" v10="-2.1951234" v11="-4.0576496"'
            ' v12="-5.7871413" v13="-8.0487804" v14="-9.379159" v15="-12.305985"'
            ' v16="-14.966741"'
            ' FilterLength="8191" InterpolateLin="0" InterpolationMethod="B-spline"'
        )
        do_command(
            'Normalize: ApplyVolume="1" PeakLevel="-1" RemoveDcOffset="1" StereoIndependent="0"'
        )

        # Step 5 – export and clean up.
        do_command(f'Export2: Filename="{output_path}" NumChannels=1')
        do_command("SelectAll:")
        do_command("RemoveTracks:")

        os.unlink(denoised_path)

        time.sleep(0.1)
       

    # Restore undo history to its default after the batch.
    do_command("SetPreference: Name=GUI/MaxUndoLevels Value=10 Reload=0")


detector = init_breath_detector()
apply_pipeline(filenames, in_dir, out_dir, detector)
