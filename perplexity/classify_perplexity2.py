# Built-In Imports
import csv
import itertools
import math
import os
from collections import defaultdict

# External Imports
import argparse
import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import tqdm

# Torch imports
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset as TorchDataset

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV

# Local Imports
from utils.featurize import normalize, t_featurize, select_features
from utils.symbolic import get_all_logprobs, get_exp_featurize, backtrack_functions
from utils.load import Dataset, get_generate_dataset

models = ["gpt"]
domains = ["wp", "reuter", "essay"]
eval_domains = ["claude", "gpt_prompt1", "gpt_prompt2", "gpt_writing", "gpt_semantic"]

if torch.cuda.is_available():
    print("Using CUDA...")
    device = torch.device("cuda")
else:
    print("Using CPU...")
    device = torch.device("cpu")

best_features_map = {}

for file in os.listdir("results"):
    if file.startswith("best_features"):
        with open(f"results/{file}") as f:
            best_features_map[file[:-4]] = f.read().strip().split("\n")

print("Loading trigram model...")
trigram_model = pickle.load(
    open("model/trigram_model.pkl", "rb"), pickle.HIGHEST_PROTOCOL
)
tokenizer = tiktoken.encoding_for_model("davinci").encode

print("Loading features...")
exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
t_data = pickle.load(open("t_data", "rb"))

print("Loading eval data...")
# exp_to_data_eval = pickle.load(open("symbolic_data_eval", "rb"))
# t_data_eval = pickle.load(open("t_data_eval", "rb"))

datasets = [
    Dataset("normal", "data/wp/human"),
    Dataset("normal", "data/wp/gpt"),
    Dataset("author", "data/reuter/human"),
    Dataset("author", "data/reuter/gpt"),
    Dataset("normal", "data/essay/human"),
    Dataset("normal", "data/essay/gpt"),
]

eval_dataset = [
    Dataset("normal", "data/wp/claude"),
    Dataset("author", "data/reuter/claude"),
    Dataset("normal", "data/essay/claude"),
    Dataset("normal", "data/wp/gpt_prompt1"),
    Dataset("author", "data/reuter/gpt_prompt1"),
    Dataset("normal", "data/essay/gpt_prompt1"),
    Dataset("normal", "data/wp/gpt_prompt2"),
    Dataset("author", "data/reuter/gpt_prompt2"),
    Dataset("normal", "data/essay/gpt_prompt2"),
    Dataset("normal", "data/wp/gpt_writing"),
    Dataset("author", "data/reuter/gpt_writing"),
    Dataset("normal", "data/essay/gpt_writing"),
    Dataset("normal", "data/wp/gpt_semantic"),
    Dataset("author", "data/reuter/gpt_semantic"),
    Dataset("normal", "data/essay/gpt_semantic"),
]

other_dataset = []

def get_scores(labels, probabilities, calibrated=False, precision=6):
    if calibrated:
        threshold = sorted(probabilities)[len(labels) - sum(labels) - 1]
    else:
        threshold = 0.5

    assert len(labels) == len(probabilities)

    if sum(labels) == 0 or sum(labels) == len(labels):
        return (
            round(accuracy_score(labels, probabilities > threshold), precision),
            round(f1_score(labels, probabilities > threshold), precision),
            -1,
        )

    return (
        round(accuracy_score(labels, probabilities > threshold), precision),
        round(f1_score(labels, probabilities > threshold), precision),
        round(roc_auc_score(labels, probabilities), precision),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--perplexity_only", action="store_true")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_file", type=str, default="results.csv")
    parser.add_argument("--save_perplexity_thres", type=str, default="perplexity_threshold.pkl")
    args = parser.parse_args()

    np.random.seed(args.seed)
    # Construct the test/train split. Seed of 0 ensures seriality across
    # all files performing the same split.
    indices = np.arange(6000)
    np.random.shuffle(indices)

    train, test = (
        indices[: math.floor(0.8 * len(indices))],
        indices[math.floor(0.8 * len(indices)) :],
    )

    # [4320 2006 5689 ... 4256 5807 4875] [5378 5980 5395 ... 1653 2607 2732]
    print("Train/Test Split:", train, test)

    # Results table, outputted to args.output_file.
    # Example Row: ["Ghostbuster (No GPT)", "WP", "gpt_wp", 0.5, 0.5, 0.5]
    results_table = [
        ["Model Type", "Experiment", "Accuracy", "F1", "AUC"],
    ]

    # Construct the generate_dataset_fn. This function takes in a featurize function,
    # which is a mapping from a file location (str) to a desired feature vector.

    generate_dataset_fn_gpt = get_generate_dataset(*datasets)
    generate_dataset_fn_eval = get_generate_dataset(*eval_dataset)

    # t_data_eval = generate_dataset_fn_eval(t_featurize, verbose=True)
    # pickle.dump(t_data_eval, open("t_data_eval", "wb"), pickle.HIGHEST_PROTOCOL)

    generate_dataset_fn = get_generate_dataset(*datasets, *eval_dataset)

    # t_data = generate_dataset_fn(t_featurize, verbose=True)
    # pickle.dump(t_data, open("t_data", "wb"), pickle.HIGHEST_PROTOCOL)

    def get_featurized_data(best_features, gpt_only=False):
        gpt_data = np.concatenate(
            [t_data] + [exp_to_data[i] for i in best_features], axis=1
        )
        if gpt_only:
            return gpt_data

        eval_data = np.concatenate(
            [t_data_eval] + [exp_to_data_eval[i] for i in best_features], axis=1
        )
        return np.concatenate([gpt_data, eval_data], axis=0)

    # Construct all indices
    def get_indices(filter_fn):
        where = np.where(generate_dataset_fn_gpt(filter_fn))[0]

        curr_train = [i for i in train if i in where]
        curr_test = [i for i in test if i in where]

        return curr_train, curr_test

    indices_dict = {}

    for model in models + ["human"]:
        train_indices, test_indices = get_indices(
            lambda file: 1 if model in file else 0,
        )

        indices_dict[f"{model}_train"] = train_indices
        indices_dict[f"{model}_test"] = test_indices

    for model in models + ["human"]:
        for domain in domains:
            train_key = f"{model}_{domain}_train"
            test_key = f"{model}_{domain}_test"

            train_indices, test_indices = get_indices(
                lambda file: 1 if domain in file and model in file else 0,
            )

            indices_dict[train_key] = train_indices
            indices_dict[test_key] = test_indices

    for key in eval_domains:
        where = np.where(generate_dataset_fn(lambda file: 1 if key in file else 0))[0]
        assert len(where) == 3000

        indices_dict[f"{key}_test"] = list(where)

    files = generate_dataset_fn(lambda x: x)
    labels = generate_dataset_fn(
        lambda file: 1 if any([m in file for m in ["gpt", "claude"]]) else 0
    )

    def train_perplexity(data, train, test, domain):
        features = data[train][:, -1].reshape(-1, 1)
        threshold = sorted(features)[len(features) - sum(labels[train]) - 1]

        pickle.dump(threshold, open(args.save_perplexity_thres, "wb"))

        probs = (data[test][:, -1] > threshold).astype(float)
        return get_scores(labels[test], probs)

    def run_experiment(best_features, model_name, train_fn, gpt_only=True):
        gpt_data = get_featurized_data(best_features, gpt_only=True)
        _, mu, sigma = normalize(gpt_data, ret_mu_sigma=True)

        data = normalize(
            get_featurized_data(best_features, gpt_only=gpt_only), mu=mu, sigma=sigma
        )

        print(f"Running {model_name} Predictions...")

        train_indices, test_indices = [], []
        for domain in domains:
            train_indices += (
                indices_dict[f"gpt_{domain}_train"]
                + indices_dict[f"human_{domain}_train"]
            )
            test_indices += (
                indices_dict[f"gpt_{domain}_test"]
                + indices_dict[f"human_{domain}_test"]
            )

            results_table.append(
                [
                    model_name,
                    f"In-Domain ({domain})",
                    *train_fn(
                        data,
                        indices_dict[f"gpt_{domain}_train"]
                        + indices_dict[f"human_{domain}_train"],
                        indices_dict[f"gpt_{domain}_test"]
                        + indices_dict[f"human_{domain}_test"],
                        "gpt",
                    ),
                ]
            )

        results_table.append(
            [
                model_name,
                "In-Domain",
                *train_fn(data, train_indices, test_indices, "gpt"),
            ]
        )

        for test_domain in domains:
            train_indices = []
            for train_domain in domains:
                if train_domain == test_domain:
                    continue

                train_indices += (
                    indices_dict[f"gpt_{train_domain}_train"]
                    + indices_dict[f"human_{train_domain}_train"]
                )

            results_table.append(
                [
                    model_name,
                    f"Out-Domain ({test_domain})",
                    *train_fn(
                        data,
                        train_indices,
                        indices_dict[f"gpt_{test_domain}_test"]
                        + indices_dict[f"human_{test_domain}_test"],
                        test_domain,
                    ),
                ]
            )

        if gpt_only:
            return

        train_indices, test_indices = [], []
        for domain in domains:
            train_indices += (
                indices_dict[f"gpt_{domain}_train"]
                + indices_dict[f"human_{domain}_train"]
            )
            test_indices += indices_dict[f"human_{domain}_test"]

        for domain in eval_domains:
            curr_test_indices = list(indices_dict[f"{domain}_test"]) + test_indices

            results_table.append(
                [
                    model_name,
                    f"Out-Domain ({domain})",
                    *train_fn(data, train_indices, curr_test_indices, "gpt"),
                ]
            )

    if args.perplexity_only:
        run_experiment(
            ["davinci-logprobs s-avg"],
            "Perplexity-Only",
            train_perplexity,
        )

    if len(results_table) > 1:
        # Write data to output csv file
        with open(args.output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerows(results_table)

        print(f"Saved results to {args.output_file}")