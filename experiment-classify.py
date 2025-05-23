# experiment-classify.py
# Author: Jorrel Rajan
#
# Args:
#   --dir <dir> : directory from which to take input examples
#   --output <output file name> : output file, csv formatted as filename, prediction


import numpy as np
import dill as pickle
import tiktoken
import openai
import argparse
import dotenv
import os
import sys
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from utils.featurize import normalize, t_featurize_logprobs, score_ngram
from utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="experiment-samples")
parser.add_argument("--openai_key", type=str, default="")
parser.add_argument("--output", type=str, default="output.csv", help="Output file for predictions")
args = parser.parse_args()

if args.openai_key != "":
    openai.api_key = args.openai_key
else:
    dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

# file = args.file
MAX_TOKENS = 2047
best_features = open("model/features.txt").read().strip().split("\n")

# Load davinci tokenizer
enc = tiktoken.encoding_for_model("davinci-002")

# Load model
model = pickle.load(open("model/model", "rb"))
mu = pickle.load(open("model/mu", "rb"))
sigma = pickle.load(open("model/sigma", "rb"))

# Train trigram
# print("Loading Trigram...")
if os.path.exists("trigram_model.pkl"):
    trigram_model = pickle.load(open("trigram_model.pkl", "rb"))
else:
    trigram_model = train_trigram() 
    pickle.dump(trigram_model, open("trigram_model.pkl", "wb"))

# Open output file
with open(args.output, 'w') as output_file:
    
    output_line = 'file, prediction'
    output_file.write(output_line + '\n')
    print(output_line)

    for file in tqdm(os.listdir(args.dir)):
        if file in ['060.txt', '074.txt', '048.txt', '049.txt', '075.txt', '061.txt', '088.txt', '077.txt', '063.txt', '062.txt', '076.txt', '089.txt', '099.txt', '072.txt', '066.txt', '067.txt', '073.txt', '098.txt', '059.txt', '065.txt', '071.txt', '070.txt', '064.txt', '058.txt', '003.txt', '017.txt', '016.txt', '002.txt', '014.txt', '028.txt', '029.txt', '001.txt', '015.txt', '039.txt', '011.txt', '005.txt', '004.txt', '010.txt', '038.txt', '006.txt', '012.txt', '013.txt', '007.txt', '022.txt', '036.txt', '037.txt', '023.txt', '035.txt', '021.txt', '009.txt', '008.txt', '020.txt', '034.txt', '018.txt', '030.txt', '024.txt']:
            continue
        file_path = os.path.join(args.dir, file)
        # Load data and featurize
        with open(file_path) as f:
            doc = f.read().strip()
            # Strip data to first MAX_TOKENS tokens
            tokens = enc.encode(doc)[:MAX_TOKENS]
            doc = enc.decode(tokens).strip()

        trigram = np.array(score_ngram(doc, trigram_model, enc.encode, n=3, strip_first=False))
        unigram = np.array(score_ngram(doc, trigram_model.base, enc.encode, n=1, strip_first=False))

        response = openai.Completion.create(
            model="babbage-002",
            prompt="<|endoftext|>" + doc,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )
        babbage = np.array(list(map(lambda x: np.exp(x), response["choices"][0]["logprobs"]["token_logprobs"][1:])))

        response = openai.Completion.create(
            model="davinci-002",
            prompt="<|endoftext|>" + doc,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )
        davinci = np.array(list(map(lambda x: np.exp(x), response["choices"][0]["logprobs"]["token_logprobs"][1:])))

        subwords = response["choices"][0]["logprobs"]["tokens"][1:]
        gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
        for i in range(len(subwords)):
            for k, v in gpt2_map.items():
                subwords[i] = subwords[i].replace(k, v)

        t_features = t_featurize_logprobs(davinci, babbage, subwords)

        vector_map = {
            "davinci-logprobs": davinci,
            "ada-logprobs": babbage,    # We have to call this ada for compatibility with the rest of the code
            "trigram-logprobs": trigram,
            "unigram-logprobs": unigram
        }

        exp_features = []
        for exp in best_features:
            exp_tokens = get_words(exp)
            curr = vector_map[exp_tokens[0]]

            for i in range(1, len(exp_tokens)):
                if exp_tokens[i] in vec_functions:
                    next_vec = vector_map[exp_tokens[i+1]]
                    curr = vec_functions[exp_tokens[i]](curr, next_vec)
                elif exp_tokens[i] in scalar_functions:
                    exp_features.append(scalar_functions[exp_tokens[i]](curr))
                    break

        data = (np.array(t_features + exp_features) - mu) / sigma
        preds = model.predict_proba(data.reshape(-1, 1).T)[:, 1]

        output_line = f"{file}, {preds}"
        # print(output_line)
        output_file.write(output_line + "\n")
