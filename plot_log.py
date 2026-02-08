import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, required=True,
                        help="Path to training log file")
    parser.add_argument("--out_prefix", type=str, default="training",
                        help="Output png prefix")
    args = parser.parse_args()

    epochs = []
    losses = []
    train_accs = []
    test_accs = []

    # ===== log 読み込み =====
    with open(args.log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("epoch"):
                continue
            e, l, tr, te = line.split(",")
            epochs.append(int(e))
            losses.append(float(l))
            train_accs.append(float(tr))
            test_accs.append(float(te))

    # ===== プロット =====
    fig, ax1 = plt.subplots()

    # 左軸：Loss
    ax1.plot(epochs, losses, label="Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # 右軸：Accuracy
    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accs, linestyle="--", label="Train Acc")
    ax2.plot(epochs, test_accs, linestyle=":", label="Test Acc")
    ax2.set_ylabel("Accuracy (%)")

    # 凡例（左右まとめる）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Training Dynamics")
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}.png")
    plt.close()

    print(f"Saved: {args.out_prefix}.png")

if __name__ == "__main__":
    main()

"""
python plot_log.py \
  --log_path log/resnet18.log \
  --out_prefix log_plot/resnet18

"""