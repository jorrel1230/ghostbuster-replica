#!/bin/bash

directories=("gpt_chosen" "human_chosen" "prompts_chosen")

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Error: Directory '$dir' not found."
        exit 1
    fi
done

# Collect all the filenames from the first directory
first_dir="${directories[0]}"
files=($(ls "$first_dir" | sort -n))

# Rename files in all directories
for dir in "${directories[@]}"; do
    i=1
    for file in "${files[@]}"; do
        if [ -f "$dir/$file" ]; then
            extension="${file##*.}"
            new_name=$(printf "%03d.%s" "$i" "$extension")
            mv "$dir/$file" "$dir/$new_name"
            echo "Renamed '$dir/$file' to '$dir/$new_name'"
            i=$((i + 1))
        else
            echo "Warning: File '$dir/$file' not found, skipping."
        fi
    done
done

echo "Renaming complete."