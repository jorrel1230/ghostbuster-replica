import argparse
import dill as pickle
import os
import numpy as np

# Local Imports
from utils.featurize import normalize, t_featurize, select_features
from utils.symbolic import get_all_logprobs, get_exp_featurize, backtrack_functions
from utils.load import Dataset, get_generate_dataset

print("Loading features...")
exp_to_data = pickle.load(open("symbolic_data_gpt", "rb"))
t_data = pickle.load(open("t_data", "rb"))

#t_data = None

def get_featurized_data(best_features, gpt_only=False):
        gpt_data = np.concatenate(
            [t_data] + [exp_to_data[i] for i in best_features], axis=1
        )
        
        return gpt_data


def process_directory(txt_dir, threshold, best_features, gpt_only=True):
    ds = [Dataset("normal", txt_dir)]
    gen_fn = get_generate_dataset(*ds)

    file_paths = gen_fn(lambda x: x)

    t_data = gen_fn(lambda x: x)

    # files = generate_dataset_fn(lambda x: x)
    """labels = generate_dataset_fn(
        lambda file: 1 if any([m in file for m in ["gpt"]]) else 0
    )"""

    gpt_data = get_featurized_data(best_features, gpt_only=True)
    _, mu, sigma = normalize(gpt_data, ret_mu_sigma=True)

    data = normalize(
        gpt_data, mu=mu, sigma=sigma
    )

    # score
    probs = (data[:, -1] > threshold).astype(float)
    print(probs)

    return [(os.path.basename(fp), p) for fp, p in zip(file_paths, probs)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--txt_dir", type=str, help="Directory containing .txt files")
    parser.add_argument("--threshold", type=str, default="perplexity_threshold.pkl")
    parser.add_argument("--output", type=str, default="perplexity_custom_results.csv")
    args = parser.parse_args()

    threshold = pickle.load(open(args.threshold, "rb"))

    results = process_directory(args.txt_dir, threshold, ["davinci-logprobs s-avg"])

    # Write results to CSV
    import csv
    with open(args.output, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ai_probability"])
        for fname, prob in results:
            writer.writerow([fname, prob])
    print(f"Saved probabilities to {args.output}")
