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

-------------------------
Version 3, current: tweaked parameters from Maé's previous version (V2): 
- steeper frequency tuning (README mentioned 90Hz, but apply_pipeline() was using 50Hz), moved RolloffType to DB24 for preserving formant quality & cut out mechanical rumble
- slowed attack to 50ms (prevents clicks)
- tuned gate to 350ms (previously at 100) to prevent chopping in-between sentence silent pauses
- lowered sensitivity tuning at 0.05 (previously 0.064) for faint backround noise in find_noise_profile_segment() & increased min_lenght to 250ms for a slightly better reduction of bird chirping(from 20, now min_lenght = 25 and also removed the imposed MIN_SILENCE_DURATION)
- denoise_audio() now has:
    prop_decrease: spectral subtraction of a 0.82 ratio (prevents clicky-sounding artifacts)
    stationary=False: adaptive tracking so the algo can adjust to noise varying in frequency (wind, humming etc)
    n_fft = 1024: finer freq bins for voice separation
- do_command() improved for error handling & prevents the script from hanging if audacity is frozen (deadline timer timeout at 10sec to kill processes if failing to respond)
   
"""

import os
import sys
import tempfile
import time
import errno
import glob
import noisereduce as nr
import soundfile as sf
import torch

# ---------------------------------------------------------------------------
# Path Configuration (Dynamic & Safe)
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR      = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
RESPIRO_PATH  = os.path.join(ROOT_DIR, "Respiro-en")

if os.path.exists(RESPIRO_PATH):
    sys.path.insert(0, RESPIRO_PATH)
else:
    print(f"ERROR: Respiro path not found at: {RESPIRO_PATH}")
    sys.exit(1)

from modules import DetectionNet, BreathDetector

# ---------------------------------------------------------------------------
# Breath Detection & Denoising Logic
# ---------------------------------------------------------------------------

def init_breath_detector():
    #Gemini's suggestion: GPU ACCELERATION: AUTOMATICALLY DETECTS CUDA TO SPEED UP NEURAL NETWORK INFERENCE ON SUPPORTED HARDWARE
    device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_file = os.path.join(RESPIRO_PATH, "respiro-en.pt")
    checkpoint      = torch.load(checkpoint_file, map_location=device, weights_only=False)
    model           = DetectionNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Respiro-EN Model Loaded on {device}")
    return BreathDetector(model, device=device)

def find_noise_profile_segment(detector, wav_path):
  
    try:
        tree = detector(wav_path, threshold=0.05, min_length=25)
        if tree:
            best = max(sorted(tree), key=lambda iv: iv.end - iv.begin)
            return best.begin, best.end
    except Exception as e:
        print(f"      Detection hint: {e}")
    return 0.0, 0.5

def denoise_audio(wav_path, noise_start, noise_end):

    data, sr  = sf.read(wav_path)
    audio     = data.T if data.ndim > 1 else data

    s_idx = int(noise_start * sr)
    e_idx = int(noise_end   * sr)
    if e_idx - s_idx < 100:
        e_idx = s_idx + 1000

    noise_sample = audio[:, s_idx:e_idx] if data.ndim > 1 else audio[s_idx:e_idx]

    reduced = nr.reduce_noise(
        y               = audio,
        sr              = sr,
        y_noise         = noise_sample,
        prop_decrease   = 0.82,
        stationary      = False,
        n_fft           = 1024,
      
    )

    reduced = reduced.T if data.ndim > 1 else reduced
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, reduced, sr)
    return tmp.name

# ---------------------------------------------------------------------------
# Audacity pipe setup - cross-platform
# ---------------------------------------------------------------------------

def get_pipes():
    """Finds Audacity pipes for Windows, Linux, or macOS."""
    if sys.platform == 'win32':
        toname = "\\\\.\\pipe\\audacity_script_pipe.to"
        fromname = "\\\\.\\pipe\\audacity_script_pipe.from"
        return (toname, fromname) if os.path.exists(toname) else (None, None)
    else:
        to_pipes = glob.glob("/tmp/audacity_script_pipe.to.*")
        from_pipes = glob.glob("/tmp/audacity_script_pipe.from.*")
        if not to_pipes or not from_pipes:
            return None, None
        return sorted(to_pipes)[-1], sorted(from_pipes)[-1]
#changed function to prevent the script running indefinitely if encountering dialog boxes
def do_command(command, timeout=10):
    """Sends command to Audacity and waits for response with improved polling."""
    POLL_INTERVAL = 0.5   
    PIPE_TIMEOUT  = 8.0   
    deadline = time.time() + PIPE_TIMEOUT
    fd = None

    while time.time() < deadline:
        toname, fromname = get_pipes()
        if toname and fromname:
            try:
                fd = os.open(toname, os.O_WRONLY | os.O_NONBLOCK)
                break
            except (FileNotFoundError, OSError):
                pass 
        time.sleep(POLL_INTERVAL)

    if fd is None:
        print("!!! Audacity pipe not found. Is Audacity open with mod-script-pipe enabled?")
        sys.exit(1)

    with os.fdopen(fd, 'w') as to_pipe:
        to_pipe.write(command + "\n")
        to_pipe.flush()

    # Read response
    _, fromname = get_pipes()
    from_fd  = os.open(fromname, os.O_RDONLY | os.O_NONBLOCK)
    read_deadline = time.time() + timeout
    with os.fdopen(from_fd, 'r') as from_pipe:
        partial = ""
        while time.time() < read_deadline:
            try:
                chunk = from_pipe.read(4096)
                if chunk:
                    partial += chunk
                    if partial.endswith("\n\n"): break
            except BlockingIOError:
                time.sleep(0.05)
    return partial.strip()

# ---------------------------------------------------------------------------
# Processing either single file/flat directories/nested directories
# ---------------------------------------------------------------------------

def process_single_file(input_path, output_path, detector):
 
    if not input_path.lower().endswith(".wav"):
        return

    # Clean up existing output to prevent Audacity "Overwrite" dialogs
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except OSError:
            print(f"  [ERROR] Cannot delete existing file: {output_path}.")
            return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 1.Respiro breath detection + noisereduce
    start, end = find_noise_profile_segment(detector, input_path)
    print(f"  [Denoise] {os.path.basename(input_path)}  profile: {start:.2f}s → {end:.2f}s")
    tmp_wav = denoise_audio(input_path, start, end)

    # 2.Audacity chain
    do_command(f'Import2: Filename="{tmp_wav}"')
    do_command("SelectAll:")
    do_command("StereoToMono:")
    do_command("SelectAll:")
    do_command('High-passFilter: Frequency=50 RolloffType="dB42"')
    do_command('NoiseGate: ATTACK="50" DECAY="350" HOLD="150" LEVEL-REDUCTION="-24" THRESHOLD="-42"')
    do_command('Normalize: ApplyVolume="1" PeakLevel="-1" RemoveDcOffset="1"')
    do_command(f'Export2: Filename="{output_path}" NumChannels=1')
    do_command("RemoveTracks:")

    if os.path.exists(output_path) and os.path.getsize(output_path) > 100:
        print(f"  [OK]  {os.path.basename(output_path)} ({os.path.getsize(output_path):,} bytes)")
    
    if os.path.exists(tmp_wav):
        os.unlink(tmp_wav)
    time.sleep(0.1)

def process_nested_folders(base_in, base_out, detector):
    """
    Auto-detection of flat vs nested layouts.
    Nested Mode: each speaker subfolder processed separately to prevent collisions.
    """
    if os.path.isfile(base_in):
        process_single_file(base_in, base_out, detector)
        return

    if not os.path.isdir(base_in):
        print(f"[ERROR] Input directory not found: {base_in}")
        return

    entries = os.listdir(base_in)
    subdirs = [e for e in entries if os.path.isdir(os.path.join(base_in, e))]
    root_wavs = [e for e in entries if e.lower().endswith(".wav")]
    
    #FLAT MODE: WAVs live directly in base_in ──────────────────────────
    if root_wavs and not subdirs:
        print(f"--- Flat layout detected ({len(root_wavs)} WAV file(s) at root) ---")
        folder_name = os.path.basename(base_in.rstrip("/"))
        out_dir = os.path.join(base_out, f"{folder_name}ok")
        for f in sorted(root_wavs):
            out_path = os.path.join(out_dir, f.replace(".wav", "ok.wav"))
            process_single_file(os.path.join(base_in, f), out_path, detector)

    #NESTED MODE, for IT: subdirectories hold the recordings ───────────────────
    elif subdirs:
        print(f"--- Nested layout detected: {len(subdirs)} subfolder(s) ---")
        for sub in sorted(subdirs):
            in_subdir = os.path.join(base_in, sub)
            out_subdir = os.path.join(base_out, f"{sub}ok")
            for f in sorted(os.listdir(in_subdir)):
                if f.lower().endswith(".wav"):
                    out_path = os.path.join(out_subdir, f.replace(".wav", "ok.wav"))
                    process_single_file(os.path.join(in_subdir, f), out_path, detector)
    else:
        print(f"[ERROR] Nothing to process in {base_in}.")

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    #PATHS HERE: set these to match your corpus directory
    IN_PATH  = "/Users/mpopa/Downloads/Androids-Corpus/Interview-Task/audio_cliptestcopie"
    OUT_PATH = "/Users/mpopa/Downloads/Androids-Corpus/Interview-Task/audio_cliptestcopieok"

    if not os.path.exists(IN_PATH):
        print(f"Input path not found: {IN_PATH}")
        sys.exit(1)

    print("--- Starting Denoise Pipeline v3 ---")
    detector = init_breath_detector()
    process_nested_folders(IN_PATH, OUT_PATH, detector)
    print("\n--- Denoising completed ---")