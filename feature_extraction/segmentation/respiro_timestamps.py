import os
import sys
import csv
import argparse
import torch
import librosa
import numpy as np
from pathlib import Path

# Configure the path to your local Respiro-en installation
# UPDATE THIS to match your local setup before running the script
RESPIRO_PATH = "/home/monicapsq/Desktop/Respiro-en"

# Ensure Respiro-en path is valid before importing modules
if os.path.exists(RESPIRO_PATH):
    sys.path.insert(0, RESPIRO_PATH)
else:
    print(f"ERROR: Respiro path not found at: {RESPIRO_PATH}")
    sys.exit(1)

# Import modules (functions in Respiro-en)
from modules import feature_extractor, DetectionNet

# Set up device and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectionNet().to(device)
checkpoint = torch.load(os.path.join(RESPIRO_PATH, 'respiro-en.pt'), map_location=device)
model.load_state_dict(checkpoint['model'])
model.eval()

# Set up parameters (using those suggested by Respiro-en paper and code)
threshold = 0.064
min_length = 20
frame_duration = 0.01  # 1 frame = 10ms for respiro-en feature_extractor stride

# Function to process a directory of .wav files and extract timestamps
def process_directory(directory_path, output_dir):
    # Find all .wav files recursively (handles both flat RT and nested IT structures)
    wav_files = [p for p in Path(directory_path).rglob('*') if p.suffix.lower() == '.wav']
    
    if not wav_files:
        print(f"No .wav files found in {directory_path}")
        return

    print(f"Found {len(wav_files)} files in {directory_path}. Processing...")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    output_csv = out_path / "timestamps.csv"

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filepath', 'start_time_sec', 'end_time_sec'])
        
        for wav_path in wav_files:
            try:
                wav, sr = librosa.load(str(wav_path), sr=16000)
                feature, length = feature_extractor(wav)
                feature, length = feature.to(device), length.to(device)
                
                with torch.no_grad():
                    output = model(feature, length)
                
                # safely handle dimension reductions to avoid errors on single elements
                prediction = (output[0] > threshold).nonzero(as_tuple=True)[0].tolist()
                
                if len(prediction) > 0:
                    diffs = np.diff(prediction)
                    splits_idx = np.where(diffs != 1)[0] + 1
                    splits = np.split(prediction, splits_idx)
                    
                    valid_splits = list(filter(lambda split: len(split) > min_length, splits))
                    
                    for split in valid_splits:
                        # Convert frame indices to seconds for SoX segmentation
                        start_time = split[0] * frame_duration
                        end_time = split[-1] * frame_duration
                        writer.writerow([str(wav_path), round(start_time, 3), round(end_time, 3)])
                        
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract breath timestamps from audio files using Respiro-en.")
    parser.add_argument("input_dir", type=str, help="Directory containing the input .wav files")
    parser.add_argument("output_dir", type=str, help="Directory to save the output timestamps.csv file")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input_dir):
        process_directory(args.input_dir, args.output_dir)
        print(f"Finished processing. Saved timestamps to {Path(args.output_dir) / 'timestamps.csv'}")
    else:
        print(f"ERROR: Directory '{args.input_dir}' not found.")