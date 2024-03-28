#!/bin/bash

# Set dataset directory
DATASET_DIR="diabetic-retinopathy-dataset"

# Function to log messages
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") $1"
}

# Function to merge and extract zip files
merge_and_extract_zip() {
    local zip_name="$1"
    log "Merging $zip_name parts into a single zip file..."
    cat "$DATASET_DIR/$zip_name".zip.* > "$DATASET_DIR/$zip_name.zip"
    log "Merged $zip_name.zip created at $DATASET_DIR"

    # Remove partition files
    rm "$DATASET_DIR/$zip_name".zip.*
    log "Removing $zip_name parts"

    # Extract the merged file
    log "Extracting $zip_name.zip..."
    unzip -o "$DATASET_DIR/$zip_name.zip" -d "$DATASET_DIR"
    log "Extracted $zip_name.zip at $DATASET_DIR"
}

# Merge and extract train.zip parts;
merge_and_extract_zip "train" &

# Merge and extract test.zip parts
merge_and_extract_zip "test" &

# End of script
log "Script execution completed."

# Wait for all background processes to finish
wait
