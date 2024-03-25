#!/bin/bash

# path where dataset will be downloaded
DATASET_DIR="data/diabetic-retinopathy-dataset"
mkdir -p "$DATASET_DIR"

# Start time of the script
start_time=$(date +%s)

# Array containing the names of the files you want to download
# Note: The files are split into multiple parts, so you need to 
# download all parts to get the complete file
# Reference: Check Data Explorer on Kaggle for the list of files
# https://www.kaggle.com/c/diabetic-retinopathy-detection/data
files=(
    "test.zip.001"
    "test.zip.002"
    "test.zip.003"
    "test.zip.004"
    "test.zip.005"
    "test.zip.006"
    "test.zip.007"
    "sampleSubmission.csv.zip"
    "sample.zip"
    "train.zip.001"
    "train.zip.002"
    "train.zip.003"
    "train.zip.004"
    "train.zip.005"
    "trainLabels.csv.zip"
)

# Define a function to download a single file
download_file() {
    kaggle competitions download -c diabetic-retinopathy-detection -f "$1" -p "$DATASET_DIR"

    local zip_file="$DATASET_DIR/$1"
    
    # If .zip extension not present in $1, append it
    if [[ "$1" != *.zip ]]; then
        zip_file="$zip_file.zip"
    fi

    # Check if zip file exists
    if [ ! -f "$zip_file" ]; then
        echo "Error: $zip_file does not exist."
        return 1
    fi

    unzip -o "$zip_file" -d "$DATASET_DIR" # -o flag to overwrite existing files
    rm -rf "$zip_file"
}

# Loop through the array of file names and download each file
for file in "${files[@]}"; do
    download_file "$file" &
done

# Wait for all background processes to finish
wait

# End time of the script
end_time=$(date +%s)

# Calculate total time taken in minutes
total_time=$(( (end_time - start_time)/60 ))

# Print total time taken
echo "Total time taken: ${total_time} minutes"
