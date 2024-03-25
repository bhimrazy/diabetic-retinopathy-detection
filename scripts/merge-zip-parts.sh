#!/bin/bash

# Set dataset directory
DATASET_DIR="diabetic-retinopathy-dataset"

# Function to log messages
log() {
    echo "$(date +"%Y-%m-%d %H:%M:%S") $1"
}

# Function to merge zip parts
merge_zip() {
    local zip_name="$1"
    log "Merging $zip_name parts into a single zip file..."
    cat "$DATASET_DIR/$zip_name".zip.* > "$DATASET_DIR/$zip_name.zip"
    log "Merged $zip_name.zip created at $DATASET_DIR"

    # remove partition files
    rm "$DATASET_DIR/$zip_name".zip.*
    log "Removing $zip_name parts"
}

# Merge train.zip parts;
merge_zip "train" &

# Merge test.zip parts
merge_zip "test" &

# End of script
log "Script execution completed."

# Wait for all background processes to finish
wait