import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.nn import functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaForSequenceClassification.from_pretrained(
    "models/roberta_wp", num_labels=2
).to(device)
roberta_model.eval()

def get_ai_probability(text):
    encoding = roberta_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = roberta_model(**encoding)
    # Probability that the label is 'AI' (class 1)
    prob = float(F.softmax(outputs.logits, dim=1)[0][1].item())
    return prob

def process_directory(txt_dir):
    results = []
    for fname in os.listdir(txt_dir):
        if fname.endswith(".txt"):
            fpath = os.path.join(txt_dir, fname)
            with open(fpath, "r") as f:
                text = f.read()
            prob = get_ai_probability(text)
            results.append((fname, prob))
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("txt_dir", help="Directory containing .txt files")
    parser.add_argument("--output", default="wp_probs.csv", help="Output CSV file")
    args = parser.parse_args()

    results = process_directory(args.txt_dir)
    # Write results to CSV
    import csv
    with open(args.output, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "ai_probability"])
        for fname, prob in results:
            writer.writerow([fname, prob])
    print(f"Saved probabilities to {args.output}")
