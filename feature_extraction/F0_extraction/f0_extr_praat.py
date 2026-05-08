import parselmouth
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def extract_f0(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Initialize a name for the output directory if not already specified
    if output_path.name != "F0_results":
        output_path = output_path / "F0_results"
        
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists() or not input_path.is_dir():
        # Check existence and validity of input directory
        print(f"Error: The input directory '{input_dir}' does not exist or is not a directory.")
        return

    # Find all subdirectories (each should represent a speaker)
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    # Check validity of structure of input directory
    if not subdirs:
        print(f"No subdirectories found in {input_dir}. Expected a nested structure.")
        return

    # Process each speaker subfolder
    for subdir in sorted(subdirs):
        speaker = subdir.name
        wave_files = sorted([f for f in subdir.glob("*.wav") if f.is_file()])
        
        if not wave_files:
            print(f"No .wav files found for speaker {speaker}. Skipping.")
            continue
            
        print(f"Processing {len(wave_files)} files for speaker: {speaker}...")
        speaker_f0_values = []

        # Output directories for this speaker
        speaker_dir = output_path / speaker
        values_dir = speaker_dir / "values"
        mean_dir = speaker_dir / "mean"
        std_dir = speaker_dir / "standard_dev"
        var_dir = speaker_dir / "variance"
        
        for d in [values_dir, mean_dir, std_dir, var_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        for file in wave_files:
            try:
                # Load audio and extract F0 using parselmouth
                sound = parselmouth.Sound(str(file))
                
                # Use 10 ms time step and 50-500 Hz range to match other F0 extraction tools' parameters
                pitch = sound.to_pitch(time_step=0.010, pitch_floor=50.0, pitch_ceiling=500.0)
                
                # Get frequency values (unvoiced frames are represented as 0.0)
                f0 = pitch.selected_array['frequency']
                times = pitch.xs()
                
                # Praat centers frames (e.g., 0.005, 0.015). We shift them to match the 0.0, 0.010 start times.
                start_times = np.maximum(np.round(times - 0.005, 3), 0.0)
                
                # Prepare and save F0 values data (time series)
                values_df = pd.DataFrame({
                    'Start_Time': start_times,
                    'End_Time': np.round(start_times + 0.010, 3),
                    'F0_Hz': f0
                })
                
                values_df.to_csv(values_dir / f"{file.stem}.csv", index=False)
                
                # Calculate statistics only on voiced frames
                voiced_f0 = f0[f0 > 0.0]
                
                u_mean = voiced_f0.mean() if len(voiced_f0) > 0 else 0.0
                u_std = voiced_f0.std() if len(voiced_f0) > 1 else 0.0
                u_var = voiced_f0.var() if len(voiced_f0) > 1 else 0.0
                
                speaker_f0_values.append({
                    "Utterance_File": file.name,
                    "F0_Mean": u_mean,
                    "F0_StdDev": u_std,
                    "F0_Variance": u_var
                })
                
            except Exception as e:
                print(f"  [Error] Processing {file.name}: {e}")

        # Save statistics for this speaker in the respective directories
        if speaker_f0_values:
            stats_df = pd.DataFrame(speaker_f0_values)
            
            stats_df[['Utterance_File', 'F0_Mean']].to_csv(mean_dir / f"{speaker}_mean.csv", index=False)
            stats_df[['Utterance_File', 'F0_StdDev']].to_csv(std_dir / f"{speaker}_standard_dev.csv", index=False)
            stats_df[['Utterance_File', 'F0_Variance']].to_csv(var_dir / f"{speaker}_variance.csv", index=False)

    print(f"\nExtraction complete!")
    print(f"Results saved in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract F0 means for nested segmented utterances.")
    parser.add_argument("input_dir", type=str, help="Directory containing speaker subfolders with .wav files")
    parser.add_argument("output_dir", type=str, help="Directory to save the resulting CSV files")
    
    args = parser.parse_args()
    extract_f0(args.input_dir, args.output_dir)
