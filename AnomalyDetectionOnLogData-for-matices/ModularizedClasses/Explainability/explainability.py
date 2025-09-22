import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _is_lstm(model) -> bool:
    shape = getattr(model, "input_shape", None)
    return isinstance(shape, tuple) and len(shape) == 3


def reconstruct(model, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct inputs via autoencoder model. Handles LSTM or dense models.

    Returns (X_recon_2d, X_2d) with shape (n_samples, n_features).
    """
    if _is_lstm(model):
        if X.ndim == 2:
            X_3d = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        else:
            X_3d = X
        recon = model.predict(X_3d, verbose=0)
        X_recon = np.reshape(recon, (recon.shape[0], recon.shape[2]))
        X_2d = np.reshape(X_3d, (X_3d.shape[0], X_3d.shape[2]))
        return X_recon, X_2d
    else:
        X_recon = model.predict(X, verbose=0)
        return X_recon, X


def per_feature_errors(X: np.ndarray, X_recon: np.ndarray, feature_names: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute per-feature squared reconstruction error and total error.
    Returns (df_feature_errors, total_error_array).
    """
    se = np.square(X - X_recon)
    total_err = np.mean(se, axis=1)
    df = pd.DataFrame(se, columns=[f"err_{f}" for f in feature_names])
    df["total_error"] = total_err
    return df, total_err


def permutation_importance_autoencoder(
    model,
    X: np.ndarray,
    feature_names: List[str],
    timesteps: int = 1,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Estimate global feature importance for an autoencoder by measuring
    the increase in reconstruction MSE when each feature is permuted.
    Works for dense and LSTM (timesteps=1 is default in this repo).
    """
    rng = np.random.default_rng(seed=random_state)

    # Baseline MSE
    X_recon, X_base = reconstruct(model, X)
    baseline_mse = np.mean(np.mean(np.square(X_base - X_recon), axis=1))

    increases = []
    for j, fname in enumerate(feature_names):
        vals = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            perm_index = rng.permutation(X_perm.shape[0])
            X_perm[:, j] = X_perm[perm_index, j]
            Xp_recon, Xp = reconstruct(model, X_perm)
            mse = np.mean(np.mean(np.square(Xp - Xp_recon), axis=1))
            vals.append(mse - baseline_mse)
        increases.append((fname, float(np.mean(vals)), float(np.std(vals))))

    df_imp = pd.DataFrame(increases, columns=["feature", "mean_increase_mse", "std"])
    df_imp.sort_values("mean_increase_mse", ascending=False, inplace=True)
    return df_imp


def _reconstruction_error_scalar(model, X: tf.Tensor) -> tf.Tensor:
    """
    Compute mean squared reconstruction error per sample as a scalar output.
    X can be 2D (n, f) or 3D (n, t, f). Returns shape (n,).
    """
    if _is_lstm(model):
        if len(X.shape) == 2:
            X_in = tf.reshape(X, (tf.shape(X)[0], 1, tf.shape(X)[1]))
        else:
            X_in = X
        recon = model(X_in, training=False)
        recon2d = tf.reshape(recon, (tf.shape(recon)[0], tf.shape(recon)[2]))
        Xin2d = tf.reshape(X_in, (tf.shape(X_in)[0], tf.shape(X_in)[2]))
        se = tf.square(Xin2d - recon2d)
    else:
        recon = model(X, training=False)
        se = tf.square(X - recon)
    return tf.reduce_mean(se, axis=1)


def integrated_gradients(
    model,
    X: np.ndarray,
    baseline: Optional[np.ndarray] = None,
    steps: int = 64,
) -> np.ndarray:
    """
    Integrated Gradients for reconstruction error scalar output.
    Returns attributions with shape (n_samples, n_features).
    """
    X = np.asarray(X, dtype=np.float32)
    n, f = X.shape[0], X.shape[-1]
    if baseline is None:
        baseline = np.zeros_like(X)
    else:
        baseline = np.broadcast_to(baseline, X.shape).astype(np.float32)

    alphas = tf.linspace(0.0, 1.0, steps + 1)  # include 0 and 1
    attributions = np.zeros((n, f), dtype=np.float32)

    for i in range(n):
        x = tf.convert_to_tensor(X[i:i+1])
        x0 = tf.convert_to_tensor(baseline[i:i+1])
        grads_sum = tf.zeros_like(x)
        for a in alphas:
            x_interp = x0 + a * (x - x0)
            with tf.GradientTape() as tape:
                tape.watch(x_interp)
                y = _reconstruction_error_scalar(model, x_interp)
            grads = tape.gradient(y, x_interp)
            grads_sum += grads
        avg_grads = grads_sum / tf.cast(steps + 1, grads_sum.dtype)
        ig = (x - x0) * avg_grads
        if _is_lstm(model):
            ig = tf.reshape(ig, (1, f))
        attributions[i] = ig.numpy().reshape(-1)
    return attributions


def global_ig_summary(
    model,
    X: np.ndarray,
    feature_names: List[str],
    out_dir: str,
    split_name: str,
    steps: int = 64,
    baseline: Optional[np.ndarray] = None,
):
    """
    Compute Integrated Gradients across a dataset and save a global summary
    (mean absolute attribution per feature) as CSV and PNG.
    """
    _ensure_dir(out_dir)
    atts = integrated_gradients(model, X, baseline=baseline, steps=steps)
    mean_abs = np.mean(np.abs(atts), axis=0)
    s = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    split_out = os.path.join(out_dir, split_name)
    _ensure_dir(split_out)
    s.to_csv(os.path.join(split_out, "integrated_gradients_global.csv"))
    plt.figure(figsize=(8, 4))
    s.plot(kind="bar")
    plt.title(f"Integrated Gradients (|attr|) – {split_name}")
    plt.ylabel("Mean |attribution|")
    plt.tight_layout()
    plt.savefig(os.path.join(split_out, "integrated_gradients_global.png"))
    plt.close()


def global_summary_from_behaviours(
    behaviours_dir: str,
    split_name: str,
    model,
    feature_names: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    n_repeats: int = 5,
):
    """
    Load processed behaviour parquet for a split (train/cv/test), compute
    global explainability artifacts:
      - Per-feature average error contribution
      - Permutation importance for reconstruction MSE
    Saves CSVs and PNG plots under out_dir/split_name.
    """
    file_map = {
        "train": "train.parquet",
        "cv": "cv.parquet",
        "test": "test.parquet",
        # detecting pipeline often uses test_processed.parquet
        "test_processed": "Test_processed.parquet",
    }
    fname = file_map.get(split_name, f"{split_name}.parquet")
    path = os.path.join(behaviours_dir, fname)
    df = pd.read_parquet(path)
    if "source" in df.columns:
        X = df.drop(columns=["source"]).values.astype("float32")
    else:
        X = df.values.astype("float32")

    if feature_names is None:
        feature_names = [
            "total_logs",
            "mean_duration",
            "fail_ratio",
            "sensitive_ratio",
            "vpn_ratio",
            "unique_patient_count",
            "unique_device_count",
            "shift_logic",
        ]

    X_recon, X_in = reconstruct(model, X)
    df_err, total_err = per_feature_errors(X_in, X_recon, feature_names)

    # Aggregate per-feature contribution as mean of squared error per feature
    contrib = df_err[[c for c in df_err.columns if c.startswith("err_")]].mean(0)
    contrib = contrib.rename(index=lambda c: c.replace("err_", ""))
    contrib_pct = contrib / contrib.sum()

    # Permutation importance
    df_perm = permutation_importance_autoencoder(model, X_in, feature_names, n_repeats=n_repeats)

    # Output
    out_dir = out_dir or os.path.join(behaviours_dir, "explainability")
    split_out = os.path.join(out_dir, split_name)
    _ensure_dir(split_out)

    contrib.sort_values(ascending=False).to_csv(os.path.join(split_out, "feature_error_contribution.csv"))
    df_perm.to_csv(os.path.join(split_out, "permutation_importance.csv"), index=False)

    # Plots
    plt.figure(figsize=(8, 4))
    contrib.sort_values(ascending=False).plot(kind="bar")
    plt.title(f"Per-feature error contribution ({split_name})")
    plt.ylabel("Mean squared error")
    plt.tight_layout()
    plt.savefig(os.path.join(split_out, "feature_error_contribution.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.bar(df_perm["feature"], df_perm["mean_increase_mse"], yerr=df_perm["std"], capsize=3)
    plt.title(f"Permutation importance (MSE increase) ({split_name})")
    plt.ylabel("Δ MSE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(split_out, "permutation_importance.png"))
    plt.close()


def anomaly_report(
    model,
    behaviours_path: str,
    raw_vectors_path: str,
    threshold: float,
    feature_names: Optional[List[str]] = None,
    out_csv: Optional[str] = None,
    top_k: int = 3,
) -> pd.DataFrame:
    """
    Build per-anomaly local explanations:
      - total_error, top-k contributing features
      - input vs reconstructed values and per-feature squared error
    Saves a flat CSV with top-k columns and returns the DataFrame.
    """
    df_beh = pd.read_parquet(behaviours_path)
    df_raw = pd.read_parquet(raw_vectors_path)

    if feature_names is None:
        feature_names = [
            "total_logs",
            "mean_duration",
            "fail_ratio",
            "sensitive_ratio",
            "vpn_ratio",
            "unique_patient_count",
            "unique_device_count",
            "shift_logic",
        ]

    X = df_beh[feature_names].values.astype("float32")
    X_recon, X_in = reconstruct(model, X)

    se = np.square(X_in - X_recon)  # (n, f)
    total_err = np.mean(se, axis=1)
    is_anom = total_err > threshold

    # Prepare flat report
    rows = []
    for i in np.where(is_anom)[0]:
        # Map to raw_df row; assume row order alignment between raw vectors and behaviours
        raw = df_raw.iloc[i] if i < len(df_raw) else None
        base = {
            "idx": int(i),
            "total_error": float(total_err[i]),
        }
        if raw is not None:
            base["UserID"] = raw.get("UserID", None)
            base["Date"] = raw.get("Date", None)
        # Top-k features by squared error
        order = np.argsort(-se[i])  # descending by error
        for rank in range(min(top_k, len(feature_names))):
            j = order[rank]
            fname = feature_names[j]
            base[f"top{rank+1}_feature"] = fname
            base[f"top{rank+1}_input"] = float(X_in[i, j])
            base[f"top{rank+1}_recon"] = float(X_recon[i, j])
            base[f"top{rank+1}_sq_error"] = float(se[i, j])
            base[f"top{rank+1}_pct"] = float(se[i, j] / (se[i].sum() + 1e-12))
        rows.append(base)

    report = pd.DataFrame(rows)

    if out_csv is None:
        out_dir = os.path.join(os.path.dirname(behaviours_path), "explainability")
        _ensure_dir(out_dir)
        out_csv = os.path.join(out_dir, "anomaly_explanations.csv")
    else:
        _ensure_dir(os.path.dirname(out_csv))

    report.to_csv(out_csv, index=False)
    return report


def global_ig_summary_from_behaviours(
    behaviours_dir: str,
    split_name: str,
    model,
    feature_names: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    steps: int = 64,
    baseline: Optional[np.ndarray] = None,
):
    """
    Convenience wrapper to compute Integrated Gradients global summary by
    loading the split parquet from behaviours_dir and writing artifacts.
    """
    file_map = {
        "train": "train.parquet",
        "cv": "cv.parquet",
        "test": "test.parquet",
        "test_processed": "Test_processed.parquet",
    }
    fname = file_map.get(split_name, f"{split_name}.parquet")
    path = os.path.join(behaviours_dir, fname)
    df = pd.read_parquet(path)
    if feature_names is None:
        feature_names = [
            "total_logs",
            "mean_duration",
            "fail_ratio",
            "sensitive_ratio",
            "vpn_ratio",
            "unique_patient_count",
            "unique_device_count",
            "shift_logic",
        ]
    X = df[feature_names].values.astype("float32")
    out_dir = out_dir or os.path.join(behaviours_dir, "explainability")
    global_ig_summary(model, X, feature_names, out_dir=out_dir, split_name=split_name, steps=steps, baseline=baseline)
