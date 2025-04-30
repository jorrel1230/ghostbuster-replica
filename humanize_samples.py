# humanize_samples.py
# Author: Jorrel Rajan
#
# Given an input directory of text files, and path to a prompt, append prompt to beginning of text file, send to gpt, and store output.

# Args:
#   --dir <dir> : directory from which to take input examples
#   --output_dir <output dir> : name of directory to put outputs in
#   --prompt_file <prompt file> : name of text file containing specific prompt to use

import openai
import argparse
import dotenv
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_essay_dir", "-i", type=str)
parser.add_argument("--input_prompt_dir", "-p", type=str)
parser.add_argument("--output_dir", "-o", type=str)
parser.add_argument("--prompt_file", "-f", type=str)


args = parser.parse_args()
dotenv.load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

# Make new dir
dir = os.mkdir(args.output_dir)

# Load Prompt file
prompt = ""
with open(args.prompt_file) as f:
    prompt = f.read().strip()

# Main Loop
for file in tqdm(os.listdir(args.input_essay_dir)):
    essay_file_path = os.path.join(args.input_essay_dir, file)

    # assumes prompt file has same name as essay file
    essay_prompt_file_path = os.path.join(args.input_prompt_dir, file) 

    # Load data and featurize
    essay_text = ""
    essay_prompt = ""

    with open(essay_file_path) as f:
        essay_text = f.read().strip()
    with open(essay_prompt_file_path) as f:
        essay_prompt = f.read().strip()

    gpt_input_prompt = f"<instruction>\n{prompt}\n</instruction>\n<original_prompt>\n{essay_prompt}\n<original_prompt>\n<original_text>\n{essay_text}\n</original_text>"

    response = openai.ChatCompletion.create(
        model="gpt-4.1-nano",
        messages=[{'role': 'user','content': gpt_input_prompt}],
    )
    
    output_doc = response['choices'][0]['message']['content']

    write_file_path = os.path.join(f"{args.output_dir}", file)
    with open(write_file_path, 'w') as f:
        f.write(output_doc)
