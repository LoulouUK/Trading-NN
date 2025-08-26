# api.py
import os, json, importlib.util
from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np

# ---- dynamic import of "trading NN.py" (no rename needed) ----
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(HERE, "trading NN.py")
spec = importlib.util.spec_from_file_location("ai_trader_module", SCRIPT_PATH)
ai = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai)  # runs imports defined in that file

app = FastAPI(title="AI Trader API", version="1.0")

# Allow opening index.html from a simple file server or other local ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly; tighten in prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    limit: int = 1000
    horizon: int = 12
    epochs: int = 12
    lr: float = 1e-3
    artifacts: str = "artifacts"

@app.get("/api/signal")
def get_signal(
    symbol: str = Query("BTCUSDT"),
    interval: str = Query("1h"),
    limit: int = Query(600, ge=200, le=2000),
    horizon: int = Query(12, ge=4, le=96),
    artifacts: str = Query("artifacts"),
    tp_rr: float = Query(1.5, ge=0.1, le=10.0),
):
    # Fetch OHLCV
    df = ai.fetch_klines_binance(symbol, interval, limit)
    # Build features (and list of feature columns used during training)
    df_feat, feature_cols = ai.build_features(df)

    # Load artifacts
    try:
        with open(os.path.join(artifacts, "scaler.json"), "r") as f:
            meta = json.load(f)
        mean = np.array(meta["mean"], dtype=np.float32)
        std = np.array(meta["std"], dtype=np.float32)
        trained_horizon = meta.get("horizon", horizon)
        # Use trained feature order if present
        if "feature_cols" in meta:
            feature_cols = meta["feature_cols"]
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Artifacts not found. Train the model first (POST /api/train).")

    X = df_feat[feature_cols].values.astype(np.float32)
    scaler = ai.Standardizer(mean=mean, std=std)
    Xn = scaler.transform(X)

    # Build & load model
    import torch
    model = ai.DualHeadNN(in_dim=len(feature_cols), hidden=64)
    try:
        model.load_state_dict(torch.load(os.path.join(artifacts, "model.pth"), map_location="cpu"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="model.pth not found. Train the model first (POST /api/train).")
    model.eval()

    x_last = torch.tensor(Xn[-1:], dtype=torch.float32)
    with torch.no_grad():
        p_buy, sl_mult = model(x_last)
    p_buy = float(p_buy.item())
    sl_mult = float(sl_mult.item())
    sl_mult = max(0.2, min(3.0, sl_mult))  # clamp

    price = float(df_feat["close"].iloc[-1])
    atr14 = float(df_feat["ATR14"].iloc[-1])

    # --- ATR fallback to avoid NaNs ---
    import math
    if atr14 is None or (isinstance(atr14, float) and math.isnan(atr14)):
        atr14 = float(df_feat["ATR14"].ffill().iloc[-1])
    if atr14 is None or (isinstance(atr14, float) and math.isnan(atr14)):
        # final fallback: 0.5% of price
        atr14 = max(1e-6, price * 0.005)

    # Trade direction (+1 long, -1 short) based on probability (discrete)
    direction_disc = 1 if p_buy >= 0.5 else -1

    # Stop-loss distance scaled by model SL multiplier and ATR
    sl_dist = sl_mult * atr14
    sl_price = float(price - sl_dist) if direction_disc == 1 else float(price + sl_dist)

    # Risk and Take-Profit using configurable risk:reward (tp_rr)
    risk = abs(price - sl_price)
    tp_price = float(price + direction_disc * tp_rr * risk)

    # Tight OHLC series for the line chart
    times = [int(ts.value // 10**6) for ts in df_feat.index]  # ms epoch
    closes = df_feat["close"].round(2).tolist()

    # --- Build a simple forward forecast path for visualization ---
    # Map interval string to minutes
    _imap = {
        '1m':1,'3m':3,'5m':5,'15m':15,'30m':30,
        '1h':60,'2h':120,'4h':240,'6h':360,'8h':480,'12h':720,
        '1d':1440,'3d':4320,'1w':10080
    }
    step_min = _imap.get(interval, 60)

    last_ts_ms = int(df_feat.index[-1].value // 10**6)
    future_t = []
    future_y = []

    # Continuous direction in [-1, 1] derived from p_buy (forecast only)
    direction_cont = max(-1.0, min(1.0, (p_buy - 0.5) * 2.0))
    # Step size between 0.25*ATR and 0.5*ATR per bar depending on confidence
    step_size = (0.25 + 0.25 * abs(direction_cont)) * atr14
    step_sign = 1.0 if direction_cont >= 0 else -1.0

    y_pred = price
    for i in range(1, int(trained_horizon) + 1):
        y_pred = y_pred + step_sign * step_size
        future_y.append(float(y_pred))
        future_t.append(last_ts_ms + i * step_min * 60_000)

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "horizon": trained_horizon,
        "price": float(price),
        "atr": float(atr14),     # preferred key used by frontend
        "atr14": float(atr14),   # backward compatibility
        "p_buy": float(p_buy),
        "sl_mult": float(sl_mult),
        "sl_price": float(sl_price),
        "tp_price": float(tp_price),
        "tp_rr": float(tp_rr),
        "direction": int(direction_disc),  # +1 long, -1 short
        "series": {"t": times, "close": closes},
        "features_used": feature_cols,
        "forecast": {"t": future_t, "y": future_y},
    }

@app.post("/api/train")
def post_train(req: TrainRequest):
    # Pull data
    df = ai.fetch_klines_binance(req.symbol, req.interval, req.limit)
    # Build features
    df_feat, feature_cols = ai.build_features(df)
    # Train (synchronous for simplicity)
    ai.train_model(
        df_feat, feature_cols,
        horizon=req.horizon,
        epochs=req.epochs,
        lr=req.lr,
        out_dir=req.artifacts
    )
    return {
        "status": "ok",
        "message": "Training completed.",
        "artifacts": req.artifacts,
        "symbol": req.symbol,
        "interval": req.interval,
        "horizon": req.horizon,
        "epochs": req.epochs,
    }

# Health endpoint
@app.get("/api/health")
async def health():
    return {"ok": True}

# Serve the frontend (index.html, app.js, styles.css) from /web at the app root
app.mount(
    "/",
    StaticFiles(directory=os.path.join(HERE, "web"), html=True),
    name="frontend",
)

# Dev entry: uvicorn api:app --reload --host 127.0.0.1 --port 8001