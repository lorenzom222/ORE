#!/bin/bash

# This script will run multiple scripts in sequence, waiting for GPU processes to finish before starting the next one.

echo "Starting full_african_linear.sh"
./eval/full_african_linear.sh
wait $(pgrep -f full_african_linear.py)

if [ $? -eq 0 ]; then
    echo "full_african_linear.sh completed successfully."
else
    echo "full_african_linear.sh failed. Exiting."
    exit 1
fi

echo "Starting full_asian_linear.sh"
./eval/full_asian_linear.sh
wait $(pgrep -f full_asian_linear.py)

if [ $? -eq 0 ]; then
    echo "full_asian_linear.sh completed successfully."
else
    echo "full_asian_linear.sh failed. Exiting."
    exit 1
fi

echo "Starting full_caucasian_linear.sh"
./eval/full_caucasian_linear.sh
wait $(pgrep -f full_caucasian_linear.py)

if [ $? -eq 0 ]; then
    echo "full_caucasian_linear.sh completed successfully."
else
    echo "full_caucasian_linear.sh failed. Exiting."
    exit 1
fi

echo "Starting full_indian_linear.sh"
./eval/full_indian_linear.sh
wait $(pgrep -f full_indian_linear.py)

if [ $? -eq 0 ]; then
    echo "full_indian_linear.sh completed successfully."
else
    echo "full_indian_linear.sh failed."
fi
