from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import UnivariateSpline

from src.config import load_config
from src.data_processing import chronological_split, load_well_data, prepare_stable_segment
from src.dataset import build_windows
from src.dca import compute_residual, fit_dca, predict_dca, reconstruct_production
from src.decomposition import train_only_eemd_features
from src.evaluation import compute_metrics, save_results
from src.features import build_features, inverse_residual_scale
from src.model import ResidualGRU
from src.training import predict_residual, train_model
from src.utils import resolve_device, set_global_seed


def run(config_path: str | Path = "config.yaml", epochs_override: int | None = None, eemd_trials_override: int | None = None) -> dict:
    root = Path(__file__).resolve().parent
    config = load_config(root / config_path)
    set_global_seed(config["training"]["seed"])
    data_cfg = config["data"]
    frame = load_well_data(
        root / data_cfg["path"], well=data_cfg["well"], input_format=data_cfg["input_format"],
        sheet_name=data_cfg.get("sheet_name", "Integrated"),
    )
    segment = prepare_stable_segment(frame, data_cfg.get("start_date"), data_cfg.get("end_date"), config["dca"]["flow_eps"])
    split, train_end = chronological_split(len(segment), **config["split"])
    time = (segment["date"] - segment["date"].iloc[0]).dt.days.to_numpy(float) / 7.0
    oil = segment["oil_rate"].to_numpy(float)
    spline_strength = float(config["dca"]["spline_strength"])
    smoothing = max(float(np.nanvar(oil[:train_end])) * train_end * spline_strength, 1e-8)
    spline = UnivariateSpline(time[:train_end], oil[:train_end], k=min(3, train_end - 1), s=smoothing)
    smooth_train = np.clip(spline(time[:train_end]), 0.0, None)
    dca_parameters = fit_dca(time[:train_end], smooth_train, config["dca"]["flow_eps"])
    dca_trend = predict_dca(time, dca_parameters)
    residual = compute_residual(oil, dca_trend)
    decomposition_cfg = dict(config["decomposition"])
    if eemd_trials_override is not None:
        decomposition_cfg["trials"] = int(eemd_trials_override)
    components, decomposition_meta = train_only_eemd_features(residual, train_end, seed=config["training"]["seed"], **decomposition_cfg)
    external = {
        name: segment[name].to_numpy(float)
        for name in ["liquid_rate", "water_rate", "gas_rate", "bottom_hole_pressure", "drawdown", "operation_time"]
    }
    features = build_features(residual, dca_trend, time, external, components, train_end)
    windows = build_windows(features.feature_array, features.target_array, split, config["model"]["lookback"])
    if len(windows.x_train) == 0 or len(windows.x_validation) == 0:
        raise ValueError("The selected well segment does not produce both training and validation windows")
    device = resolve_device(config["training"]["device"])
    model = ResidualGRU(
        input_dim=features.feature_array.shape[1], hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"], dropout=config["model"]["dropout"],
    )
    epochs = int(epochs_override or config["training"]["epochs"])
    model, loss_history = train_model(
        model, windows.x_train, windows.y_train, windows.x_validation, windows.y_validation,
        epochs=epochs, batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"], device=device,
    )
    train_scaled = predict_residual(model, windows.x_train, device)
    validation_scaled = predict_residual(model, windows.x_validation, device)
    indices = np.concatenate([windows.idx_train, windows.idx_validation])
    predicted_residual = inverse_residual_scale(np.concatenate([train_scaled, validation_scaled]), features.scalers)
    predicted_production = reconstruct_production(dca_trend[indices], predicted_residual)
    prediction_rows = pd.DataFrame({
        "time": segment["date"].iloc[indices].dt.strftime("%Y-%m-%d").to_numpy(),
        "actual_production": oil[indices], "dca_trend": dca_trend[indices],
        "actual_residual": residual[indices], "predicted_residual": predicted_residual,
        "predicted_production": predicted_production, "dataset_split": split[indices],
    })
    validation_mask = prediction_rows["dataset_split"].eq("validation").to_numpy()
    metrics = {
        "validation_production": compute_metrics(prediction_rows.loc[validation_mask, "actual_production"], prediction_rows.loc[validation_mask, "predicted_production"]),
        "validation_residual": compute_metrics(prediction_rows.loc[validation_mask, "actual_residual"], prediction_rows.loc[validation_mask, "predicted_residual"]),
        "dca_parameters": dca_parameters.to_dict(),
        "data_protocol": "chronological 80% training / 20% validation; no test set",
        "prediction_semantics": "one-step observed-history diagnostic validation",
        "residual_definition": "residual = q_DCA - q_actual",
        "reconstruction_definition": "q_pred = q_DCA - residual_pred",
    }
    resolved = {
        **config, "training": {**config["training"], "epochs": epochs},
        "feature_names": features.feature_names, "feature_shape": list(features.feature_array.shape),
        "train_end": train_end, "decomposition_meta": decomposition_meta,
        "scaler_fit_scope": "train_only", "dca_fit_scope": "train_only",
    }
    paths = save_results(root / config["output"]["results_dir"], prediction_rows, metrics, loss_history, resolved, model)
    print(f"segment_rows={len(segment)} train_end={train_end}")
    print(f"X_train={windows.x_train.shape} X_validation={windows.x_validation.shape}")
    print(f"feature_count={len(features.feature_names)} device={device} epochs={epochs}")
    print(f"validation_metrics={metrics['validation_production']}")
    for label, path in paths.items():
        print(f"{label}: {path.relative_to(root)}")
    return {"metrics": metrics, "paths": paths, "feature_names": features.feature_names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the single-well EDCA-PR-DLF example")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Optional smoke-test override")
    parser.add_argument("--eemd-trials", type=int, default=None, help="Optional smoke-test override")
    args = parser.parse_args()
    run(args.config, args.epochs, args.eemd_trials)
