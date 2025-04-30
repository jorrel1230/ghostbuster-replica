#!/bin/bash

# Create directories if they don't exist
mkdir -p "./gpt_chosen"
mkdir -p "./human_chosen"
mkdir -p "./prompts_chosen"

# Read the list of files from output.txt
while IFS= read -r file; do
    # Copy files from ./gpt to ./gpt_chosen
    if [ -f "./gpt/$file" ]; then
        cp "./gpt/$file" "./gpt_chosen/$file"
    else
        echo "File ./gpt/$file not found"
    fi

    # Copy files from ./human to ./human_chosen
    if [ -f "./human/$file" ]; then
        cp "./human/$file" "./human_chosen/$file"
    else
        echo "File ./human/$file not found"
    fi

    # Copy files from ./prompts to ./prompts_chosen
    if [ -f "./prompts/$file" ]; then
        cp "./prompts/$file" "./prompts_chosen/$file"
    else
        echo "File ./prompts/$file not found"
    fi
done < output.txt

echo "File copying complete."