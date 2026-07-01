import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from power_forecast.data import inverse_target
from power_forecast.models import build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_scaled(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_loss += float(loss.item()) * len(xb)
            n += len(xb)
    return total_loss / max(1, n)


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in loader:
            pred = model(xb.to(device)).cpu().numpy()
            outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"mse": mse, "mae": mae}


def train_one_run(
    window_data,
    model_name: str,
    horizon: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device_name: str,
    checkpoint_path: Path | None = None,
) -> dict:
    set_seed(seed)
    device = torch.device(device_name)

    model = build_model(
        model_name=model_name,
        n_features=window_data.x_train.shape[-1],
        horizon=horizon,
    ).to(device)

    train_loader = make_loader(window_data.x_train, window_data.y_train, batch_size, True)
    val_loader = make_loader(window_data.x_val, window_data.y_val, batch_size, False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, patience // 3)
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict[str, float]] = []

    progress = tqdm(range(1, epochs + 1), desc=f"{model_name}-h{horizon}-s{seed}")
    for epoch in progress:
        model.train()
        train_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(xb)
            n += len(xb)

        train_loss /= max(1, n)
        val_loss = evaluate_scaled(model, val_loader, device, criterion)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        progress.set_postfix(train=f"{train_loss:.4f}", val=f"{val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    scaled_pred = predict(model, window_data.x_test, device, batch_size)
    y_pred = inverse_target(scaled_pred, window_data.target_scaler)
    y_true = inverse_target(window_data.y_test, window_data.target_scaler)
    metrics = compute_metrics(y_true, y_pred)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_name": model_name,
                "horizon": horizon,
                "seed": seed,
                "metrics": metrics,
            },
            checkpoint_path,
        )

    return {
        "model": model_name,
        "horizon": horizon,
        "seed": seed,
        "epochs_ran": len(history),
        "best_val_loss_scaled": best_val,
        "metrics": metrics,
        "history": history,
        "y_true": y_true,
        "y_pred": y_pred,
        "test_dates": window_data.test_dates,
    }


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
