import requests
from tqdm import tqdm
import os
import argparse
import dotenv
import json

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="experiment-samples")
parser.add_argument("--output", type=str, default="output.json", help="Output file for predictions")
args = parser.parse_args()

dotenv.load_dotenv()
gptzero_apikey = os.getenv("GPTZERO_API_KEY")

if not gptzero_apikey:
    raise ValueError("GPTZERO_API_KEY not in env")

MAX_TOKENS = 2047

headers = {
    "x-api-key": gptzero_apikey,
    "Content-Type": "application/json",
    "Accept": "application/json"
}

url = "https://api.gptzero.me/v2/predict/text"

all_results = []

for file in tqdm(os.listdir(args.dir)):
    if file in ['060.txt', '074.txt', '048.txt', '049.txt', '075.txt', '061.txt', '088.txt', '077.txt', '063.txt', '062.txt', '076.txt', '089.txt', '099.txt', '072.txt', '066.txt', '067.txt', '073.txt', '098.txt', '059.txt', '065.txt', '071.txt', '070.txt', '064.txt', '058.txt', '003.txt', '017.txt', '016.txt', '002.txt', '014.txt', '028.txt', '029.txt', '001.txt', '015.txt', '039.txt', '011.txt', '005.txt', '004.txt', '010.txt', '038.txt', '006.txt', '012.txt', '013.txt', '007.txt', '022.txt', '036.txt', '037.txt', '023.txt', '035.txt', '021.txt', '009.txt', '008.txt', '020.txt', '034.txt', '018.txt', '030.txt', '024.txt']:
        continue
    file_path = os.path.join(args.dir, file)
    # Load data and featurize
    with open(file_path) as f:
        doc = f.read().strip()[:MAX_TOKENS]

    """output_line = f"{file}, {preds}"
    # print(output_line)
    output_file.write(output_line + "\n")"""

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

    print(f"{file}, {result['documents'][0]['average_generated_prob']}")

with open(args.output, 'w') as output_file:
    json.dump(all_results, output_file, indent=2)