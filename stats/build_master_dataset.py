import os
import glob
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def parse_speaker_dir(speaker_dir_name):
    """
    Parses the speaker directory name (e.g., 'IT_01_CF56_1_segmented' or 'RT_01_CF56_1ok_segmented')
    to extract metadata.
    """
    if not speaker_dir_name.endswith("_segmented"):
        return None
    
    prefix = speaker_dir_name.replace("_segmented", "")
    full_recording_id = prefix
    
    is_denoised = 0
    if "ok" in prefix.lower():
        is_denoised = 1
        prefix = prefix.replace("ok", "").replace("OK", "")
        
    task = "Unknown"
    recording_id = prefix
    if prefix.startswith("IT_"):
        task = "IT"
        recording_id = prefix[3:]
    elif prefix.startswith("RT_"):
        task = "RT"
        recording_id = prefix[3:]
    else:
        if "IT" in prefix:
            task = "IT"
        elif "RT" in prefix:
            task = "RT"
        recording_id = prefix.replace("IT_", "").replace("RT_", "").replace("IT", "").replace("RT", "").strip("_")

    # Extract metadata from recording_id (e.g., '01_CF56_1')
    parts = recording_id.split("_")
    if len(parts) >= 3:
        try:
            speaker_id = str(int(parts[0]))
        except ValueError:
            speaker_id = parts[0]
            
        part1 = parts[1]
        speaker_grp = part1[0] if len(part1) > 0 else "Unknown"
        sex = part1[1] if len(part1) > 1 else "Unknown"
        try:
            age = int(part1[2:])
        except ValueError:
            age = np.nan
    else:
        speaker_grp = "Unknown"
        sex = "Unknown"
        age = np.nan
        speaker_id = "Unknown"

    age_grp = "Unknown"
    if not pd.isna(age):
        age_grp = ">=47" if age >= 47 else "<47"

    return {
        "Recording_ID": full_recording_id,
        "Speaker_ID": speaker_id,
        "Age": age,
        "Age_grp": age_grp,
        "Sex": sex,
        "Speaker_grp": speaker_grp,
        "Task": task,
        "is_denoised": is_denoised
    }

def build_dataset(input_dir, output_dir):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Error: {input_dir} does not exist.")
        return

    print(f"Searching for F0 statistics in {input_dir}...")
    
    # Recursively find all mean CSV files
    mean_files = list(input_path.rglob("*_mean.csv"))
    if not mean_files:
        print("No mean CSV files found in the specified directory.")
        return

    dataset_rows = []

    for mean_csv in mean_files:
        speaker_dir_name = mean_csv.parent.parent.name
        info = parse_speaker_dir(speaker_dir_name)
        if not info:
            continue
        
        # Get Mean F0
        mean_f0 = np.nan
        try:
            df_mean = pd.read_csv(mean_csv)
            if 'F0_Mean' in df_mean.columns:
                # Average the utterance means to get the recording mean
                mean_f0 = df_mean['F0_Mean'].mean()
        except Exception as e:
            print(f"Error reading {mean_csv}: {e}")

        # Get Standard Deviation F0
        sd_csv = mean_csv.parent.parent / "standard_dev" / mean_csv.name.replace("_mean.csv", "_standard_dev.csv")
        sd_f0 = np.nan
        if sd_csv.exists():
            try:
                df_sd = pd.read_csv(sd_csv)
                if 'F0_StdDev' in df_sd.columns:
                    # Average the utterance standard deviations
                    sd_f0 = df_sd['F0_StdDev'].mean()
            except Exception as e:
                print(f"Error reading {sd_csv}: {e}")

        # Get Variance F0
        var_csv = mean_csv.parent.parent / "variance" / mean_csv.name.replace("_mean.csv", "_variance.csv")
        var_f0 = np.nan
        if var_csv.exists():
            try:
                df_var = pd.read_csv(var_csv)
                if 'F0_Variance' in df_var.columns:
                    var_f0 = df_var['F0_Variance'].mean()
            except Exception as e:
                print(f"Error reading {var_csv}: {e}")

        row = info.copy()
        row["Mean_F0"] = mean_f0
        row["F0_SD"] = sd_f0
        dataset_rows.append(row)

    df_out = pd.DataFrame(dataset_rows)
    if df_out.empty:
        print("No valid data extracted to build the dataset.")
        return

    # Define standard column order
    columns_order = [
        "Recording_ID", "Speaker_ID", "Age", "Age_grp", "Sex", "Speaker_grp", 
        "Task", "is_denoised", "Mean_F0", "F0_SD"
    ]
    
    # Keep any extra columns safely
    for col in df_out.columns:
        if col not in columns_order:
            columns_order.append(col)
            
    df_out = df_out[columns_order]
    
    # Sort the dataset for better readability
    df_out.sort_values(by=["Recording_ID", "Task", "is_denoised"], inplace=True)
    
    out_path = Path(output_dir) / "master_dataset.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Dataset successfully compiled and saved to {out_path} with {len(df_out)} records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build master dataset from pre-computed F0 extraction results.")
    parser.add_argument("input_dir", type=str, help="Path to the F0 extraction results folder (e.g., F0_results/F0_egemaps)")
    parser.add_argument("output_dir", type=str, help="Directory to save the output dataset CSV file")
    
    args = parser.parse_args()
    build_dataset(args.input_dir, args.output_dir)
