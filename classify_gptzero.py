import requests
from tqdm import tqdm
import os
import argparse
import dotenv
import json
import tiktoken

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="experiment-samples")
parser.add_argument("--output", type=str, default="output.json", help="Output file for predictions")
args = parser.parse_args()

dotenv.load_dotenv()
gptzero_apikey = os.getenv("GPTZERO_API_KEY")

if not gptzero_apikey:
    raise ValueError("GPTZERO_API_KEY not in env")

MAX_TOKENS = 2047

# Load davinci tokenizer
enc = tiktoken.encoding_for_model("davinci-002")

headers = {
    "x-api-key": gptzero_apikey,
    "Content-Type": "application/json",
    "Accept": "application/json"
}

url = "https://api.gptzero.me/v2/predict/text"

all_results = []

for file in tqdm(os.listdir(args.dir)):
    file_path = os.path.join(args.dir, file)
    # Load data and featurize
    with open(file_path, encoding='utf-8') as f:
        doc = f.read().strip()
        # Strip data to first MAX_TOKENS tokens (done here for text length consistency with ghostbuster experiment)
        tokens = enc.encode(doc)[:MAX_TOKENS]
        doc = enc.decode(tokens).strip()

    payload = {
        "document": doc,
        "multilingual": False
    }
    
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"Error for file {file}: {response.status_code}")

    result = response.json()

    all_results.append({
        "file": file,
        "response": result
    })

    #print(f"{file}, {result['documents'][0]['average_generated_prob']}")

with open(args.output, 'w') as output_file:
    json.dump(all_results, output_file, indent=2)