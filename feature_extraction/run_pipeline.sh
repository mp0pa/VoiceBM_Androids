#!/bin/bash

# Default values
F0_METHOD="praat"
RAW_AUDIO=""
OUTPUT_BASE=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --raw-audio) RAW_AUDIO="$2"; shift ;;
        --output-dir) OUTPUT_BASE="$2"; shift ;;
        --f0-method) F0_METHOD="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$RAW_AUDIO" ] || [ -z "$OUTPUT_BASE" ]; then
    echo "Usage: $0 --raw-audio <raw_audio_dir> --output-dir <base_output_dir> [--f0-method egemaps|yin|praat]"
    exit 1
fi

echo "Starting Full Audio Processing Pipeline..."
mkdir -p "$OUTPUT_BASE"

echo "========================================"
echo "Step 1: Timestamp Detection (Respiro-EN)"
echo "========================================"
python feature_extraction/segmentation/respiro_timestamps.py "$RAW_AUDIO" "$OUTPUT_BASE"

if [ ! -f "$OUTPUT_BASE/timestamps.csv" ]; then
    echo "Error: timestamps.csv was not generated. Aborting."
    exit 1
fi

echo "========================================"
echo "Step 2: Segmentation (SoX)"
echo "========================================"
SEGMENTED_DIR="$OUTPUT_BASE/segmented_clips"
python feature_extraction/segmentation/segment.py "$RAW_AUDIO" "$OUTPUT_BASE/timestamps.csv" "$SEGMENTED_DIR"

echo "========================================"
echo "Step 3: F0 Extraction ($F0_METHOD)"
echo "========================================"
F0_OUTPUT_DIR="$OUTPUT_BASE"

if [ "$F0_METHOD" == "egemaps" ]; then
    python feature_extraction/F0_extraction/f0_extr_egemaps.py "$SEGMENTED_DIR" "$F0_OUTPUT_DIR"
elif [ "$F0_METHOD" == "yin" ]; then
    python feature_extraction/F0_extraction/f0_extr_yin.py "$SEGMENTED_DIR" "$F0_OUTPUT_DIR"
elif [ "$F0_METHOD" == "praat" ]; then
    python feature_extraction/F0_extraction/f0_extr_praat.py "$SEGMENTED_DIR" "$F0_OUTPUT_DIR"
else
    echo "Error: Invalid F0 method. Choose from: egemaps, yin, praat"
    exit 1
fi

echo "========================================"
echo "Pipeline complete! All results are in: $OUTPUT_BASE"