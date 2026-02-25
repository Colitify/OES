"""Train LSTM temporal predictor on BOSCH OES data.

Loads BOSCH NetCDF data, computes PCA(20) temporal embedding, trains an
LSTM to predict the next embedding vector from a 10-step history, and saves
the trained model + train/val loss curve.

Acceptance: validation MSE < initial MSE (model learns something).

Usage:
    python scripts/train_temporal.py --data data/bosch_oes/ --epochs 50
    python scripts/train_temporal.py --data data/bosch_oes/ --epochs 100 --seq_len 10
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(
        description="LSTM next-step predictor on BOSCH OES PCA embedding"
    )
    parser.add_argument("--data", type=str, default="data/bosch_oes/",
                        help="Directory containing BOSCH NetCDF files")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (default 50)")
    parser.add_argument("--seq_len", type=int, default=10,
                        help="LSTM sequence length / history window (default 10)")
    parser.add_argument("--n_components", type=int, default=20,
                        help="PCA components for temporal embedding (default 20)")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="LSTM hidden state size (default 64)")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="LSTM layers (default 2)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default 1e-3)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Mini-batch size (default 64)")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation fraction (default 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_model", type=str, default=None,
                        help="Output model path (default: outputs/lstm_temporal.pt)")
    parser.add_argument("--out_plot", type=str, default=None,
                        help="Output loss curve path (default: outputs/lstm_loss.png)")
    args = parser.parse_args()

    import torch
    from src.data_loader import load_bosch_oes
    from src.temporal import LSTMPredictor, compute_temporal_embedding, train_lstm

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print(f"Loading BOSCH OES data from {args.data}...")
    data = load_bosch_oes(args.data)
    spectra = data["spectra"]
    print(f"  Loaded: {spectra.shape} spectra")

    # PCA embedding
    print(f"Computing PCA({args.n_components}) temporal embedding...")
    embedding, pca = compute_temporal_embedding(spectra, n_components=args.n_components)
    print(f"  Embedding: {embedding.shape}")

    # Build and train LSTM
    print(f"\nTraining LSTM({args.hidden_size}, {args.n_layers} layers) "
          f"for {args.epochs} epochs, seq_len={args.seq_len}...")
    model = LSTMPredictor(
        n_features=args.n_components,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        dropout=0.1,
    )
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

    history = train_lstm(
        model, embedding,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        batch_size=args.batch_size,
    )

    train_losses = history["train_loss"]
    val_losses = history["val_loss"]

    print(f"\n  Initial val MSE:  {val_losses[0]:.6f}")
    print(f"  Final val MSE:    {val_losses[-1]:.6f}")
    print(f"  Initial train MSE:{train_losses[0]:.6f}")
    print(f"  Final train MSE:  {train_losses[-1]:.6f}")

    # Acceptance check: val MSE decreases
    val_improved = val_losses[-1] < val_losses[0]
    status = "PASS" if val_improved else "FAIL"
    improvement = (val_losses[0] - val_losses[-1]) / val_losses[0] * 100
    print(f"\n  Val MSE improvement: {improvement:.1f}% → {status}")

    # Save model
    model_path = Path(args.out_model) if args.out_model else Path("outputs/lstm_temporal.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_features": args.n_components,
        "hidden_size": args.hidden_size,
        "n_layers": args.n_layers,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "pca_components": pca.components_.astype(np.float32),
        "pca_mean": pca.mean_.astype(np.float32) if pca.mean_ is not None else None,
    }, str(model_path))
    print(f"  Saved model to {model_path}")

    # Plot loss curve
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(range(1, args.epochs + 1), train_losses,
                label="Train MSE", color="steelblue", lw=1.5)
    ax.semilogy(range(1, args.epochs + 1), val_losses,
                label="Val MSE", color="tomato", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title(
        f"LSTM Temporal Predictor — Train/Val Loss\n"
        f"epochs={args.epochs}, seq_len={args.seq_len}, "
        f"hidden={args.hidden_size}, n_layers={args.n_layers}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = Path(args.out_plot) if args.out_plot else Path("outputs/lstm_loss.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved loss curve to {plot_path}")

    print(f"\n{'='*50}")
    print(f"Val MSE decreased: {val_improved} → {status}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
