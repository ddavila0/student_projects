import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_throughput(model_data, site, path):
    cols = [c for c in model_data.columns if c.startswith("throughput_") and c != "total_throughput"]
    fig, ax = plt.subplots(figsize=(14, 6))
    for col in cols:
        ax.plot(model_data.index, model_data[col], label=col.replace("throughput_", ""),
                linewidth=1.2, alpha=0.8)
    ax.plot(model_data.index, model_data["total_throughput"], label="total",
            color="black", linewidth=2.2, linestyle="--")
    ax.set_title(f"Throughput by source site -> {site}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Throughput (bytes/s)")
    ax.legend(title="Source site", loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weights(weights, site, path):
    fig, ax = plt.subplots(figsize=(12, 6))
    for b in weights.index:
        ax.plot(weights.columns, weights.loc[b] / 1e6, marker="o", label=b, linewidth=1.5)
        ax.axhline(weights.loc[b].mean() / 1e6, color="gray", linewidth=0.5,
                   linestyle=":", alpha=0.5)
    ax.set_ylim(0, None)
    ax.set_title(f"Weight per CV fold -> {site}")
    ax.set_xlabel("CV fold")
    ax.set_ylabel("weight (MB/s per active transfer)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)