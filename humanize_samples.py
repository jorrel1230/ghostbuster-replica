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

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--prompt_file", type=str)


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
for file in os.listdir(args.dir):
    file_path = os.path.join(args.dir, file)
    # Load data and featurize
    with open(file_path) as f:
        doc = f.read().strip()

        """
        # call gpt endpoint
        response = client.responses.create(
            model="gpt-4.1-nano",
            input=[
                {"role": "user", "content": f'<instruction>\n{prompt}\n</instruction>\n<original_text>\n{doc}\n</original_text>'}
            ],
            text={
                "format": {
                    "type": "text"
                }
            },
            reasoning={},
            tools=[],
            temperature=1,
            max_output_tokens=2048,
            top_p=1,
            store=False
        )
        """

        response = openai.ChatCompletion.create(
            model="gpt-4.1-nano",
            messages=[{'role': 'user','content': f'<instruction>\n{prompt}\n</instruction>\n<original_text>\n{doc}\n</original_text>'}],
        )
        
        output_doc = response['choices'][0]['message']['content']

        write_file_path = os.path.join(f"{args.output_dir}", file)
        with open(write_file_path, 'w') as f:
            f.write(output_doc)
