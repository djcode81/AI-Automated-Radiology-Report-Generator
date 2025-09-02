#!/bin/bash
# Script to run the CT scan analysis pipeline from command line

# Check if input file was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_file.nii.gz> [output_report.txt]"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-${INPUT_FILE%.nii.gz}_report.txt}"

# Run the pipeline
cd "$(dirname "$0")"
python pipeline.py "$INPUT_FILE" --output "$OUTPUT_FILE"
