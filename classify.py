import numpy as np
import dill as pickle
import tiktoken
import openai
import argparse

from sklearn.linear_model import LogisticRegression
from utils.featurize import normalize, t_featurize_logprobs, score_ngram
from utils.symbolic import train_trigram, get_words, vec_functions, scalar_functions

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, default="input.txt")
parser.add_argument("--openai_key", type=str, default="")
args = parser.parse_args()

if args.openai_key != "":
    openai.api_key = args.openai_key

file = args.file
MAX_TOKENS = 2047
best_features = open("model/features.txt").read().strip().split("\n")

# Load davinci tokenizer
enc = tiktoken.encoding_for_model("gpt-4o-mini")

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

trigram_model = train_trigram()

# Convert subwords to token IDs for trigram scoring
response = openai.ChatCompletion.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": doc}],
    logprobs=True,
    temperature=0.0,
)

subwords = [x["token"] for x in response["choices"][0]["logprobs"]["content"][1:]]

gpt2_map = {"\n": "Ċ", "\t": "ĉ", " ": "Ġ"}
for i in range(len(subwords)):
    for k, v in gpt2_map.items():
        subwords[i] = subwords[i].replace(k, v)

ada = np.array(list(map(lambda x: np.exp(x["logprob"]), response["choices"][0]["logprobs"]["content"][1:])))
davinci = ada

# Calculate trigram and unigram scores using subwords
trigram_scores = []
unigram_scores = []

# Add padding tokens for trigram context
padded_subwords = [50256, 50256] + subwords

# Score each position
for i in range(len(subwords)):
    trigram_context = padded_subwords[i:i+3]
    unigram_context = [padded_subwords[i+2]]
    
    trigram_scores.append(trigram_model.n_gram_probability(trigram_context))
    unigram_scores.append(trigram_model.base.n_gram_probability(unigram_context))

trigram = np.array(trigram_scores)
unigram = np.array(unigram_scores)

print(ada.shape)
print(davinci.shape) 
print(unigram.shape)
print(trigram.shape)
print(subwords.__len__())

t_features = t_featurize_logprobs(davinci, ada, subwords)

vector_map = {
    "davinci-logprobs": davinci,
    "ada-logprobs": ada,
    "trigram-logprobs": trigram,
    "unigram-logprobs": unigram
}

exp_features = []
for exp in best_features:

    exp_tokens = get_words(exp)
    curr = vector_map[exp_tokens[0]]

    for i in range(1, len(exp_tokens) - 1):  # Adjusted to prevent index out of range
        if exp_tokens[i] in vec_functions:
            next_vec = vector_map[exp_tokens[i + 1]]
            curr = vec_functions[exp_tokens[i]](curr, next_vec)
        elif exp_tokens[i] in scalar_functions:
            exp_features.append(scalar_functions[exp_tokens[i]](curr))
            break

# Ensure the last token is processed if it's a scalar function
if exp_tokens[-1] in scalar_functions:
    exp_features.append(scalar_functions[exp_tokens[-1]](curr))

print("-=-=-=-")
print(len(t_features))
print(len(exp_features))
print(mu.shape)
print(sigma.shape)

data = (np.array(t_features + exp_features) - mu) / sigma
preds = model.predict_proba(data.reshape(-1, 1).T)[:, 1]

print(f"Prediction: {preds}")
