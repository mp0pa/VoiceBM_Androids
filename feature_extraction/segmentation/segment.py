import os
import csv
import argparse
from pathlib import Path
import sox

def segment_audio(audio_dir, csv_file, output_dir):
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found.")
        return

    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {csv_file}...")
    
    # Group breath intervals by their source file
    file_intervals = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            orig_path = Path(row['filepath'])
            
            # Smart path resolution for flexibility
            direct_path = Path(audio_dir) / orig_path.name
            nested_path = Path(audio_dir) / orig_path
            
            if direct_path.exists():
                filepath = str(direct_path)
            elif nested_path.exists():
                filepath = str(nested_path)
            elif orig_path.exists():
                filepath = str(orig_path)
            else:
                filepath = str(direct_path)  # Default for error reporting
            
            start = float(row['start_time_sec'])
            end = float(row['end_time_sec'])
            
            if filepath not in file_intervals:
                file_intervals[filepath] = []
            file_intervals[filepath].append((start, end))
            
    # Process each file
    for filepath, breath_intervals in file_intervals.items():
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        try:
            # Need the total duration to get the final utterance slice at the end of the file
            duration = sox.file_info.duration(filepath)
        except Exception as e:
            print(f"Could not read duration for {filepath}: {e}")
            continue
        
        # Sort breath intervals chronologically to be safe
        breath_intervals.sort(key=lambda x: x[0])
        
        # Invert the breath intervals to get speech utterances
        utterance_intervals = []
        current_time = 0.0
        
        for b_start, b_end in breath_intervals:
            # The gap between 'current_time' and the next breath is an utterance
            if b_start > current_time:
                utterance_intervals.append((current_time, b_start))
            current_time = b_end
        
        # Add the final utterance from the last breath to the end of the file
        if current_time < duration:
            utterance_intervals.append((current_time, duration))
        
        # Define naming and folder structure
        orig_path = Path(filepath)
        # Create a dedicated directory for the segments of this specific file
        out_dir = out_base / f"{orig_path.stem}_segmented"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Segmenting {orig_path.name} into {len(utterance_intervals)} utterances (saving to {out_dir}/)...")
        
        # 5. Extract using pysox
        for idx, (u_start, u_end) in enumerate(utterance_intervals):
            # Filter out spurious segments (< 0.2 seconds)
            if (u_end - u_start) < 0.2:
                continue
                
            out_filename = f"{orig_path.stem}_utt{idx+1:03d}.wav"
            out_filepath = out_dir / out_filename
            
            tfm = sox.Transformer()
            tfm.trim(u_start, u_end)
            try:
                tfm.build_file(str(orig_path), str(out_filepath))
            except Exception as e:
                print(f"    [Error] PySox failed to write {out_filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment audio files into utterances using a breath timestamps CSV.")
    parser.add_argument("audio_dir", type=str, help="Directory containing the input audio files")
    parser.add_argument("csv_file", type=str, help="Path to the timestamps CSV file")
    parser.add_argument("output_dir", type=str, help="Directory to save segmented folders")
    
    args = parser.parse_args()
    segment_audio(args.audio_dir, args.csv_file, args.output_dir)