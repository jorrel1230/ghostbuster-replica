#!/bin/bash

# Initialize an empty array to store text files and selected files
text_files=(*.txt)
selected_files=()

# Shuffle the text files randomly
for i in "${!text_files[@]}"; do
    j=$((RANDOM % (i + 1)))
    temp="${text_files[i]}"
    text_files[i]="${text_files[j]}"
    text_files[j]="$temp"
done

# Iterate through shuffled text files
for file in "${text_files[@]}"; do
    # Check if file exists to prevent errors if no .txt files
    if [[ -f "$file" ]]; then
        # Print the contents of the file
        echo "File: $file"
        cat "$file"
        
        # Prompt user for selection
        echo ""
        read -p "Save this file? (y/n): " choice
        
        # Check user's choice
        if [[ "$choice" == "y" ]]; then
            selected_files+=("$file")
            
            # Stop if we've reached 100 selected files
            if [[ "${#selected_files[@]}" -eq 100 ]]; then
                break
            fi
        fi
    fi
done

# Print out all selected files
echo "Selected files:" > output.txt
for selected_file in "${selected_files[@]}"; do
    echo "$selected_file" >> output.txt
done
