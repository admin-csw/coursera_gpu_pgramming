#!/bin/bash
set -e

INPUT="${1:-data/Lena.pgm}"

if [ ! -f "$INPUT" ]; then
    echo "Error: Input file '$INPUT' not found."
    echo "Usage: ./run.sh [input_image.pgm]"
    exit 1
fi

if [ ! -f bin/imageRotationNPP ]; then
    echo "Binary not found. Building..."
    make
fi

echo "Running image rotation on: $INPUT"
./bin/imageRotationNPP --input "$INPUT"
