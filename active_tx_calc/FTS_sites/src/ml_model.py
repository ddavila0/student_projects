import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


def fit_weights(model_data, buckets, cfg):
    m = cfg["model"]
    kf = KFold(n_splits=m["n_splits"], shuffle=True, random_state=m["random_state"])
    results, weights_per_fold = [], []

    for i, (tr_i, te_i) in enumerate(kf.split(model_data)):
        train, test = model_data.iloc[tr_i], model_data.iloc[te_i]
        w = pd.Series({b: (train[f"throughput_{b}"].sum() / train[f"active_transfers_{b}"].sum()
                           if train[f"active_transfers_{b}"].sum() > 0 else 0.0)
                       for b in buckets})
        weights_per_fold.append(w.rename(f"fold_{i}"))

        y_hat = sum(test[f"active_transfers_{b}"] * w[b] for b in buckets)
        y_t = test["total_throughput"]
        results.append({"fold": i, "R2": r2_score(y_t, y_hat),
                        "WAPE": (y_hat - y_t).abs().sum() / y_t.sum()})

    cv = pd.DataFrame(results).set_index("fold")
    weights = pd.concat(weights_per_fold, axis=1)
    weights.index.name = "source_site"
    return cv, weights