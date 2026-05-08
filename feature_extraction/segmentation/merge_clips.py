import os
import argparse
import sox
from pathlib import Path
import wave
import shutil

def create_silence_wav(filepath, duration=0.5, sr=16000, channels=1):
    """Creates a completely silent WAV file of the specified duration."""
    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2) # 16-bit
        wf.setframerate(sr)
        wf.writeframes(b'\x00' * (2 * channels * int(duration * sr)))

def merge_speaker_files(input_dir, output_dir):
    input_path = Path(input_dir)
    
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: The input directory '{input_dir}' does not exist or is not a directory.")
        return
        
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all subdirectories (each represents a speaker)
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"No subdirectories found in {input_dir}. Expected a nested structure (e.g., Speaker1/utt1.wav).")
        return
        
    for subdir in sorted(subdirs):
        speaker_name = subdir.name
        # Gather and sort all wav files chronologically in the speaker's folder
        wav_files = sorted([str(f) for f in subdir.glob("*.wav")])
        
        if not wav_files:
            print(f"No .wav files found for speaker {speaker_name}. Skipping.")
            continue
            
        out_filepath = output_path / f"{speaker_name}.wav"
        
        if len(wav_files) == 1:
            print(f"Only 1 file found for {speaker_name}. Copying directly to {out_filepath.name}...")
            shutil.copy2(wav_files[0], out_filepath)
            continue
            
        print(f"Merging {len(wav_files)} files for {speaker_name} into {out_filepath.name}...")
        
        silence_path = None
        try:
            # Dynamically match the sample rate and channels of the input files
            sr = int(sox.file_info.sample_rate(wav_files[0]))
            channels = int(sox.file_info.channels(wav_files[0]))
            
            silence_path = output_path / f"temp_silence_{speaker_name}.wav"
            create_silence_wav(silence_path, duration=0.5, sr=sr, channels=channels)
            
            # Interleave the silence file between actual utterances
            files_to_merge = []
            for i, f in enumerate(wav_files):
                files_to_merge.append(f)
                if i < len(wav_files) - 1:
                    files_to_merge.append(str(silence_path))

            combiner = sox.Combiner()
            combiner.build(files_to_merge, str(out_filepath), 'concatenate')
        except Exception as e:
            print(f"  [Error] Failed to merge files for {speaker_name}: {e}")
        finally:
            if silence_path and silence_path.exists():
                silence_path.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple utterance files per speaker into a single audio file.")
    parser.add_argument("input_dir", type=str, help="Directory containing speaker subdirectories with .wav files")
    parser.add_argument("output_dir", type=str, help="Directory to save the merged .wav files")
    
    args = parser.parse_args()
    merge_speaker_files(args.input_dir, args.output_dir)
