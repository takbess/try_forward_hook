import matplotlib.pyplot as plt

epochs, steps, means, vars_ = [], [], [], []

with open("feat_stats.log") as f:
    for line in f:
        e, s, m, v = line.strip().split(",")
        epochs.append(int(e))
        means.append(float(m))
        vars_.append(float(v))

plt.figure()
plt.plot(means)
plt.xlabel("Iteration")
plt.ylabel("Feature Mean")
plt.title("Feature Mean over Training")
plt.savefig("feature_mean.png")

plt.figure()
plt.plot(vars_)
plt.xlabel("Iteration")
plt.ylabel("Feature Variance")
plt.title("Feature Variance over Training")
plt.savefig("feature_variance.png")
