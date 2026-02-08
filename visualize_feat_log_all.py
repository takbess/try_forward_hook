import matplotlib.pyplot as plt

# ===== log ファイル一覧 =====
log_files = {
    "ResNet18": "log_feat_stats/resnet18.log",
    # "ResNet34": "log_feat_stats/resnet34.log",
    # "ResNet18 + KD": "log_feat_stats/kd.log",
    # "ResNet18 + CrossKD": "log_feat_stats/crosskd.log",
}

# ===== 読み込み関数 =====
def load_stats(path):
    means, vars_ = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("epoch"):
                continue
            _, _, mean, var = line.split(",")
            means.append(float(mean))
            vars_.append(float(var))
    return means, vars_

# ===== Mean plot =====
plt.figure()
for label, path in log_files.items():
    means, _ = load_stats(path)
    plt.plot(means, label=label)

plt.xlabel("Iteration")
plt.ylabel("Feature Mean")
plt.title("Feature Mean during Training")
plt.legend()
plt.tight_layout()
plt.savefig("log_feat_stats/mean.png")
plt.close()

# ===== Variance plot =====
plt.figure()
for label, path in log_files.items():
    _, vars_ = load_stats(path)
    plt.plot(vars_, label=label)

plt.xlabel("Iteration")
plt.ylabel("Feature Variance")
plt.title("Feature Variance during Training")
plt.legend()
plt.tight_layout()
plt.savefig("log_feat_stats/variance.png")
plt.close()

print("Saved: log_feat_stats/mean.png, log_feat_stats/variance.png")
