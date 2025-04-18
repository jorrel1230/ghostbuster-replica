import numpy as np
import dill as pickle
import tiktoken
import openai
import argparse
import dotenv
import os

from sklearn.linear_model import LogisticRegression
from utils.featurize import normalize, t_featurize_logprobs, score_ngram
from utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="input.txt")
parser.add_argument("--openai_key", type=str, default="")
args = parser.parse_args()

if args.openai_key != "":
    openai.api_key = args.openai_key
else:
    dotenv.load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

file = args.file
MAX_TOKENS = 2047
best_features = open("model/features.txt").read().strip().split("\n")

# Load davinci tokenizer
enc = tiktoken.encoding_for_model("davinci-002")

# Load model
model = pickle.load(open("model/model", "rb"))
mu = pickle.load(open("model/mu", "rb"))
sigma = pickle.load(open("model/sigma", "rb"))

# Load data and featurize
with open(file) as f:
    doc = f.read().strip()
    # Strip data to first MAX_TOKENS tokens
    tokens = enc.encode(doc)[:MAX_TOKENS]
    doc = enc.decode(tokens).strip()

    print(f"Input: {doc}")

# Train trigram
print("Loading Trigram...")

if os.path.exists("trigram_model.pkl"):
    trigram_model = pickle.load(open("trigram_model.pkl", "rb"))
else:
    trigram_model = train_trigram()
    pickle.dump(trigram_model, open("trigram_model.pkl", "wb"))


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

# print("t_features.shape", len(t_features))
# print("davinci.shape", davinci.shape)
# print("babbage.shape", babbage.shape)
# print("trigram.shape", trigram.shape)
# print("unigram.shape", unigram.shape)

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

print(f"Prediction: {preds}")