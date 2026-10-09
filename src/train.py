import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import layers
import model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data_split/data_training.csv")
    p.add_argument("--valid_dataset", default="data_split/data_validation.csv")
    p.add_argument("--layer", type=int, nargs="+", default=[24, 24, 24])
    p.add_argument("--epochs", type=int, default=84)
    p.add_argument("--loss", default="binaryCrossentropy")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.0314)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_path", default="saved_model.npy")
    p.add_argument("--figures_dir", default="results/figures")
    return p.parse_args()


def load_split(path):
    df = pd.read_csv(path, header=None)
    y_raw = df.iloc[:, 1].values
    x = df.iloc[:, 2:].values.astype(np.float64)
    y = (y_raw == "M").astype(int)
    return x, y


def plot_learning_curves(history, out_dir):
    epochs = range(1, len(history["loss"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(epochs, history["loss"], label="training loss")
    ax[0].plot(epochs, history["val_loss"], label="validation loss", linestyle="--")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Learning Curves")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(epochs, history["acc"], label="training acc")
    ax[1].plot(epochs, history["val_acc"], label="validation acc")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Learning Curves")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "learning_curves.png")
    plt.savefig(path, dpi=150)
    print(f"> learning curves saved to '{path}'")
    plt.show()


def main():
    args = parse_args()

    x_train, y_train = load_split(args.dataset)
    x_val, y_val = load_split(args.valid_dataset)

    print(f"x_train shape : {x_train.shape}")
    print(f"x_valid shape : {x_val.shape}")

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-12
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    hidden_specs = [
        layers.DenseLayer(u, activation="sigmoid", weights_initializer="heUniform")
        for u in args.layer
    ]
    output_spec = layers.DenseLayer(2, activation="softmax",
                                    weights_initializer="heUniform")
    network = model.createNetwork(x_train.shape[1],
                                  hidden_specs + [output_spec],
                                  seed=args.seed)

    history = model.fit(
        network,
        (x_train, y_train),
        (x_val, y_val),
        loss=args.loss,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )

    print(f"> saving model '{args.save_path}' to disk...")
    model.save(network, args.save_path,
               norm_stats={"mean": mean, "std": std})

    plot_learning_curves(history, args.figures_dir)


if __name__ == "__main__":
    main()