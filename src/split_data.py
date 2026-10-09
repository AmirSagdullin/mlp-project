import argparse
import os
import numpy as np
import pandas as pd

HEADER = (
    ["id", "diagnosis"]
    + ["mean radius", "mean texture", "mean perimeter", "mean area",
       "mean smoothness", "mean compactness", "mean concavity",
       "mean concave points", "mean symmetry", "mean fractal dimension"]
    + [f"{n} error" for n in
       ["radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave points", "symmetry",
        "fractal dimension"]]
    + ["worst radius", "worst texture", "worst perimeter", "worst area",
       "worst smoothness", "worst compactness", "worst concavity",
       "worst concave points", "worst symmetry", "worst fractal dimension"]
)


def stratified_split(df, valid_size, seed):
    rng = np.random.default_rng(seed)
    idx_b = df.index[df["diagnosis"] == "B"].to_numpy().copy()
    idx_m = df.index[df["diagnosis"] == "M"].to_numpy().copy()
    rng.shuffle(idx_b)
    rng.shuffle(idx_m)

    n_b_val = int(round(len(idx_b) * valid_size))
    n_m_val = int(round(len(idx_m) * valid_size))

    val_idx = np.concatenate([idx_b[:n_b_val], idx_m[:n_m_val]])
    train_idx = np.concatenate([idx_b[n_b_val:], idx_m[n_m_val:]])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return df.loc[train_idx].reset_index(drop=True), \
           df.loc[val_idx].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/data.csv")
    p.add_argument("--out_dir", default="data_split")
    p.add_argument("--valid_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.data, header=None, names=HEADER)
    train_df, valid_df = stratified_split(df, args.valid_size, args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    train_df.to_csv(f"{args.out_dir}/data_training.csv", index=False, header=False)
    valid_df.to_csv(f"{args.out_dir}/data_validation.csv", index=False, header=False)

    for name, d in [("Train", train_df), ("Valid", valid_df)]:
        b = int((d["diagnosis"] == "B").sum())
        m = int((d["diagnosis"] == "M").sum())
        print(f"{name}: shape={d.shape}  B={b} ({b/len(d)*100:.1f}%)  "
              f"M={m} ({m/len(d)*100:.1f}%)")


if __name__ == "__main__":
    main()