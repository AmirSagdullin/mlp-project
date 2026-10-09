import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data_split/data_validation.csv")
    p.add_argument("--model_path", default="saved_model.npy")
    return p.parse_args()


def load_split(path):
    df = pd.read_csv(path, header=None)
    y_raw = df.iloc[:, 1].values
    x = df.iloc[:, 2:].values.astype(np.float64)
    y = (y_raw == "M").astype(int)
    return x, y


def main():
    args = parse_args()

    x, y = load_split(args.dataset)
    network, norm_stats = model.load(args.model_path)

    if norm_stats is not None:
        x = (x - norm_stats["mean"]) / norm_stats["std"]

    proba = model.predict_proba(network, x)
    preds = np.argmax(proba, axis=1)

    loss = model.binary_crossentropy_loss(network, x, y)
    acc = model.accuracy_score(network, x, y)

    print(f"Test samples: {len(y)}")
    print(f"Binary cross-entropy: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")

    n_show = min(10, len(y))
    print("\nFirst predictions:")
    for i in range(n_show):
        label = "M" if preds[i] == 1 else "B"
        true = "M" if y[i] == 1 else "B"
        mark = "ok" if preds[i] == y[i] else "MISS"
        print(f"  #{i:02d}: pred={label} true={true} [{mark}]  p(M)={proba[i, 1]:.3f}")


if __name__ == "__main__":
    main()