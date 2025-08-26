#!/usr/bin/env python3
"""
AI Trader — Dual‑Head Neural Network that fuses 10 trading strategies
====================================================================

What this script does
---------------------
• Pulls OHLCV data from the **Binance Spot public REST API** (no API key required).
• Engineers features from **10 popular strategies** (incl. Bollinger Bands, EMA cross, RSI, MACD, Stoch, Donchian, OBV, TD‑Sequential style MRI surrogate, Fair Value Gaps, Liquidity Sweeps) + ATR.
• Trains a compact PyTorch model with two heads:
    1) Classification head → P(BUY) within horizon H (hit +0.5×ATR before −0.5×ATR).
    2) Regression head → predicted **stop‑loss distance** (in ATR multiples) for the next H bars (adverse excursion).
• Runs live inference: prints **entry price** (last close), **probability**, and **suggested SL**.

IMPORTANT
---------
This is **educational** code, not financial advice. Markets are risky. Past performance ≠ future results. You are responsible for any use.

Quickstart (local)
------------------
1) Python 3.10+
2) `pip install numpy pandas requests torch`
3) Train (example, 1h BTC):
   `python ai_trader.py --mode train --symbol BTCUSDT --interval 1h --limit 1000 --horizon 12`
4) Live signal:
   `python ai_trader.py --mode live  --symbol BTCUSDT --interval 1h --limit 600 --horizon 12`

You can change `--symbol` to e.g. ETHUSDT, and `--interval` to 15m, 1h, 4h, 1d (Binance intervals).

"MRI" note: the proprietary Momentum Reversal Indicator (MRI) isn’t public. We include a **TD‑Sequential‑style setup & exhaustion proxy** that captures similar exhaustion dynamics.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests

# PyTorch is required for this script. Install with: pip install torch
try:
    import sys
    import torch
    import torch.nn as nn
    import torch.optim as optim
    # Helpful banner so you can verify interpreter + torch at runtime
    print(f"[AI Trader] Python: {sys.executable}")
    print(f"[AI Trader] Torch : {getattr(torch, '__version__', 'unknown')}")
except Exception as e:  # pragma: no cover
    import subprocess, sys as _sys
    try:
        pip_ver = subprocess.check_output([_sys.executable, '-m', 'pip', '--version']).decode(errors='ignore')
    except Exception:
        pip_ver = '(pip not available)'
    raise ImportError(
        "PyTorch is required. Install it inside the SAME interpreter/venv you're using to run this script.\n"
        f"Interpreter: {_sys.executable}\n"
        f"pip version: {pip_ver}\n"
        f"Original error: {e}"
    )

BINANCE_BASE = "https://api.binance.com"

INTERVALS = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "8h": "8h", "12h": "12h",
    "1d": "1d", "3d": "3d", "1w": "1w"
}


# -------------------------------
# Data fetching (Binance: /api/v3/klines)
# -------------------------------
def fetch_klines_binance(symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    if interval not in INTERVALS:
        raise ValueError(f"Unsupported interval {interval}")
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": INTERVALS[interval], "limit": int(limit)}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(data, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.set_index("close_time", inplace=True)
    return df


# -------------------------------
# Technical indicators (vectorized)
# -------------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1)
    return ranges.max(axis=1)

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=period, adjust=False).mean()
    roll_down = down.ewm(span=period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-12)
    d = k.rolling(window=d_period).mean()
    return k, d

def bollinger(series: pd.Series, window: int = 20, n_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    mid = sma(series, window)
    sd = series.rolling(window).std()
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    width = (upper - lower) / (mid + 1e-12)
    pctb = (series - lower) / ((upper - lower) + 1e-12)
    return mid, upper, lower, width, pctb

def donchian(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    upper = df["high"].rolling(window).max()
    lower = df["low"].rolling(window).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0))
    return (direction * df["volume"]).fillna(0).cumsum()

# TD‑Sequential style setup/exhaustion proxy (MRI‑like)
# Count consecutive closes above/below close[4] back → normalize 0..1

def td_setup_strength(close: pd.Series, lookback: int = 4, max_count: int = 9) -> pd.Series:
    bull = (close > close.shift(lookback)).astype(int)
    bear = (close < close.shift(lookback)).astype(int)
    count = pd.Series(0, index=close.index, dtype=float)
    for i in range(len(close)):
        if i == 0:
            count.iat[i] = 0
        else:
            if bull.iat[i]:
                count.iat[i] = max(1, count.iat[i-1] + 1) if count.iat[i-1] >= 0 else 1
            elif bear.iat[i]:
                count.iat[i] = min(-1, count.iat[i-1] - 1) if count.iat[i-1] <= 0 else -1
            else:
                count.iat[i] = 0
        # clamp
        if count.iat[i] > max_count:
            count.iat[i] = max_count
        if count.iat[i] < -max_count:
            count.iat[i] = -max_count
    return (count / max_count).astype(float)

# Fair Value Gap (ICT 3‑candle) detection
# Bullish FVG at t: low[t] > high[t-2]; Bearish FVG at t: high[t] < low[t-2]

def fair_value_gap(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    bull_gap = (df["low"] > df["high"].shift(2)).astype(int)
    bear_gap = (df["high"] < df["low"].shift(2)).astype(int)
    bull_size = (df["low"] - df["high"].shift(2)).clip(lower=0)
    bear_size = (df["low"].shift(2) - df["high"]).clip(lower=0)
    return (bull_gap * bull_size), (bear_gap * bear_size)

# Liquidity sweep: take out prior N‑bar high/low but close back within prior range (wicky)

def liquidity_sweep(df: pd.DataFrame, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
    prev_high = df["high"].rolling(lookback).max().shift(1)
    prev_low = df["low"].rolling(lookback).min().shift(1)
    swept_high = ((df["high"] > prev_high) & (df["close"] < prev_high)).astype(int)
    swept_low  = ((df["low"] < prev_low) & (df["close"] > prev_low)).astype(int)
    return swept_high, swept_low


# -------------------------------
# Feature engineering
# -------------------------------

def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    out["ATR14"] = atr(out, 14)

    # 1) Bollinger
    mid, up, lo, bb_width, bb_pctb = bollinger(out["close"], 20, 2)
    out["BB_width"] = bb_width
    out["BB_pctB"] = bb_pctb

    # 2) EMA cross
    out["EMA12"] = ema(out["close"], 12)
    out["EMA26"] = ema(out["close"], 26)
    out["EMA_diff"] = (out["EMA12"] - out["EMA26"]) / (out["ATR14"] + 1e-12)

    # 3) RSI
    out["RSI14"] = rsi(out["close"], 14)

    # 4) MACD
    macd_line, signal_line, hist = macd(out["close"], 12, 26, 9)
    out["MACD"] = macd_line
    out["MACD_sig"] = signal_line
    out["MACD_hist"] = hist

    # 5) Stochastic
    k, d = stochastic(out, 14, 3)
    out["STO_K"] = k
    out["STO_D"] = d

    # 6) Donchian
    d_hi, d_lo, d_mid = donchian(out, 20)
    out["Don_pos"] = (out["close"] - d_lo) / (d_hi - d_lo + 1e-12)

    # 7) OBV
    out["OBV"] = obv(out)
    out["OBV_chg"] = out["OBV"].diff().fillna(0)

    # 8) TD‑style MRI proxy
    out["TD_setup"] = td_setup_strength(out["close"], 4, 9)

    # 9) Fair Value Gaps
    bull_fvg, bear_fvg = fair_value_gap(out)
    out["FVG_bull"] = bull_fvg / (out["ATR14"] + 1e-12)
    out["FVG_bear"] = bear_fvg / (out["ATR14"] + 1e-12)

    # 10) Liquidity sweeps
    sweep_high, sweep_low = liquidity_sweep(out, 20)
    out["Sweep_high"] = sweep_high
    out["Sweep_low"] = sweep_low

    # Normalize some features
    out["RET_1"] = out["close"].pct_change()

    feature_cols = [
        "BB_width", "BB_pctB",
        "EMA_diff",
        "RSI14",
        "MACD", "MACD_sig", "MACD_hist",
        "STO_K", "STO_D",
        "Don_pos",
        "OBV_chg",
        "TD_setup",
        "FVG_bull", "FVG_bear",
        "Sweep_high", "Sweep_low",
        "RET_1",
        "ATR14"
    ]
    out = out.dropna().copy()
    return out, feature_cols


# -------------------------------
# Labeling
# -------------------------------

def make_labels(df: pd.DataFrame, horizon: int = 12, tp_atr: float = 0.5, sl_atr: float = 0.5) -> Tuple[pd.Series, pd.Series]:
    """
    BUY label (1) if **within next H bars** price first reaches +tp_atr*ATR above entry before reaching -sl_atr*ATR.
    Else SELL/0 if the opposite happens. If neither is touched, label is NaN (ignored in training).

    Regression target: adverse excursion (in ATR multiples) in next H bars → used as SL predictor.
    """
    close = df["close"].values
    atrv = df["ATR14"].values
    highs = df["high"].rolling(horizon).max().shift(-horizon+1)
    lows  = df["low"].rolling(horizon).min().shift(-horizon+1)

    # Fast vectorized approximation for label
    tp_prices = df["close"] + tp_atr * df["ATR14"]
    sl_prices = df["close"] - sl_atr * df["ATR14"]
    future_high = highs
    future_low  = lows

    buy = (future_high >= tp_prices) & (future_low > sl_prices)
    sell = (future_low <= sl_prices) & (future_high < tp_prices)

    y_cls = pd.Series(np.where(buy, 1.0, np.where(sell, 0.0, np.nan)), index=df.index)

    # adverse excursion
    future_min_low = lows
    adverse = (df["close"] - future_min_low) / (df["ATR14"] + 1e-12)
    y_sl = adverse.clip(lower=0.0, upper=3.0)

    return y_cls, y_sl


# -------------------------------
# Dataset utilities
# -------------------------------
@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / (self.std + 1e-12)


# -------------------------------
# Model
# -------------------------------
class DualHeadNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.head_cls = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.head_sl  = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus(beta=1.0))  # positive

    def forward(self, x):
        h = self.backbone(x)
        p = self.head_cls(h)
        sl = self.head_sl(h)  # ATR multiples
        return p.squeeze(-1), sl.squeeze(-1)


# -------------------------------
# Training / Inference
# -------------------------------

def train_model(df: pd.DataFrame, feature_cols: List[str], horizon: int = 12, epochs: int = 12, lr: float = 1e-3,
                out_dir: str = "artifacts"):
    if torch is None:
        raise RuntimeError("PyTorch not available. Install torch to train.")

    y_cls, y_sl = make_labels(df, horizon=horizon)

    # Keep only labeled rows (where y_cls is not NaN)
    mask = y_cls.notna()
    dfx = df.loc[mask]
    y_cls = y_cls.loc[mask]
    y_sl = y_sl.loc[mask]

    X = dfx[feature_cols].values.astype(np.float32)
    y1 = y_cls.values.astype(np.float32)
    y2 = y_sl.values.astype(np.float32)

    # Standardize (z‑score)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    scaler = Standardizer(mean=mean, std=std)
    Xn = scaler.transform(X).astype(np.float32)

    # Train/validation split
    n = len(Xn)
    n_train = int(n * 0.8)
    Xtr, Xva = Xn[:n_train], Xn[n_train:]
    y1tr, y1va = y1[:n_train], y1[n_train:]
    y2tr, y2va = y2[:n_train], y2[n_train:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualHeadNN(in_dim=Xn.shape[1], hidden=64).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    # Class imbalance handling
    pos_weight = max(1.0, float((1 - y1tr.mean()) / (y1tr.mean() + 1e-6)))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    # But our head already has Sigmoid; switch to BCELoss.
    bce = nn.BCELoss(weight=None)
    mse = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr, device=device)
    y1tr_t = torch.tensor(y1tr, device=device)
    y2tr_t = torch.tensor(y2tr, device=device)
    Xva_t = torch.tensor(Xva, device=device)
    y1va_t = torch.tensor(y1va, device=device)
    y2va_t = torch.tensor(y2va, device=device)

    best_va = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        p_tr, sl_tr = model(Xtr_t)
        loss = bce(p_tr, y1tr_t) + 0.2 * mse(sl_tr, y2tr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p_va, sl_va = model(Xva_t)
            va_loss = bce(p_va, y1va_t) + 0.2 * mse(sl_va, y2va_t)
            acc = ((p_va.cpu().numpy() > 0.5) == (y1va > 0.5)).mean() if len(y1va) else float('nan')
        print(f"Epoch {epoch:02d} | train={loss.item():.4f} val={va_loss.item():.4f} val_acc={acc:.3f}")
        if va_loss.item() < best_va:
            best_va = va_loss.item()
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pth"))
            with open(os.path.join(out_dir, "scaler.json"), "w") as f:
                json.dump({"mean": mean.tolist(), "std": std.tolist(), "feature_cols": feature_cols, "horizon": horizon}, f)


@torch.no_grad()
def live_signal(df: pd.DataFrame, feature_cols: List[str], artifacts: str = "artifacts"):
    if torch is None:
        raise RuntimeError("PyTorch not available. Install torch to run inference.")
    # Load artifacts
    with open(os.path.join(artifacts, "scaler.json"), "r") as f:
        meta = json.load(f)
    mean = np.array(meta["mean"], dtype=np.float32)
    std = np.array(meta["std"], dtype=np.float32)
    horizon = meta.get("horizon", 12)

    model = DualHeadNN(in_dim=len(feature_cols), hidden=64)
    model.load_state_dict(torch.load(os.path.join(artifacts, "model.pth"), map_location="cpu"))
    model.eval()

    # Compute features on provided df
    df_feat, fcols = build_features(df)
    X = df_feat[feature_cols].values.astype(np.float32)
    scaler = Standardizer(mean=mean, std=std)
    Xn = scaler.transform(X)

    x_last = torch.tensor(Xn[-1:], dtype=torch.float32)
    p_buy, sl_mult = model(x_last)

    p_buy = float(p_buy.item())
    sl_mult = float(sl_mult.item())

    # Clamp sensible range
    sl_mult = max(0.2, min(3.0, sl_mult))

    price = float(df_feat["close"].iloc[-1])
    atr14 = float(df_feat["ATR14"].iloc[-1])
    sl_price = price - sl_mult * atr14

    print("— Live Signal —")
    print(f"Close: {price:.2f} | ATR14: {atr14:.2f}")
    print(f"P(BUY next {horizon} bars): {p_buy:.3f}")
    print(f"Suggested entry: {price:.2f} | Suggested SL: {sl_price:.2f}  (−{sl_mult:.2f}×ATR)")


# -------------------------------
# CLI
# -------------------------------

def main():
    ap = argparse.ArgumentParser("AI Trader — dual‑head NN")
    ap.add_argument("--mode", choices=["train", "live"], required=True)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    # Fetch data
    df = fetch_klines_binance(args.symbol, args.interval, args.limit)
    df_feat, feature_cols = build_features(df)

    if args.mode == "train":
        train_model(df_feat, feature_cols, horizon=args.horizon, epochs=args.epochs, lr=args.lr, out_dir=args.artifacts)
    else:
        live_signal(df, feature_cols, artifacts=args.artifacts)


if __name__ == "__main__":
    main()
