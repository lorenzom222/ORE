#!/bin/bash

# This script will run multiple scripts in sequence, waiting for GPU processes to finish before starting the next one.

echo "Starting full_african_skewed.sh"
./full_african_skewed.sh
wait $(pgrep -f full_african_skewed.py)

if [ $? -eq 0 ]; then
    echo "full_african_skewed.sh completed successfully."
else
    echo "full_african_skewed.sh failed. Exiting."
    exit 1
fi

echo "Starting full_cauasian_skewed.sh"
./full_cauasian_skewed.sh
wait $(pgrep -f full_cauasian_skewed.py)

if [ $? -eq 0 ]; then
    echo "full_cauasian_skewed.sh completed successfully."
else
    echo "full_cauasian_skewed.sh failed. Exiting."
    exit 1
fi

echo "Starting full_asian_skewed.sh"
./full_asian_skewed.sh
wait $(pgrep -f full_asian_skewed.py)

if [ $? -eq 0 ]; then
    echo "full_asian_skewed.sh completed successfully."
else
    echo "full_asian_skewed.sh failed. Exiting."
    exit 1
fi

echo "Starting full_indian_skewed.sh"
./full_indian_skewed.sh
wait $(pgrep -f full_indian_skewed.py)

if [ $? -eq 0 ]; then
    echo "full_indian_skewed.sh completed successfully."
else
    echo "full_indian_skewed.sh failed."
fi
