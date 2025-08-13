#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Intraday Options Bot — Hybrid TA + AI (paper trading via robin_stocks quotes)

What’s new in this build
- RTH/market-day gate with a simple toggle (ENFORCE_RTH)
- Correct CT market close (3:00 PM), weekend/holiday awareness
- True 1‑minute prefilter + 5‑minute confirm path
- VWAP resets each session (no multi-day bleed)
- TA/AI thresholds actually respected
- AI cooldown + per-cycle cap
- Post‑TA logging of potential trades (CSV + console)
- Post‑AI logging of Top 3 (CSV + console)
- Safer CSV header logic; removed INIT row hack
- Fixed JSON build (consistent indicators in recco), misc robustness

State files
- open_trades.json (open state)
- closed_trades.csv (exits log)
- last_recommendations.json (latest picks)
- logs/ta_candidates.csv (TA pass list, appended)
- logs/ai_top3.csv (Top 3 AI-confirmed per cycle, appended)
"""

import os
import json
import time
import math
import csv
import sys
from datetime import datetime, timedelta, date
from dateutil import tz, parser as dateparser
from dotenv import load_dotenv
from tqdm import tqdm
import contextlib, builtins
import logging
import schedule

import numpy as np
import pandas as pd
import yfinance as yf

# robin_stocks for quotes (paper)
import robin_stocks.robinhood as r

load_dotenv()

# ── AI (optional) ────────────────────────────────────────────────────────────
USE_AI = True  # set False to disable AI blending
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

try:
    if USE_AI and OPENAI_API_KEY:
        from openai import OpenAI
        _client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        _client = None
except Exception:
    _client = None
    USE_AI = False

# ─────────────────────────────────────────────────────────────────────────────
# 0) USER SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSES = {
    "core30": [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","LLY","V","JPM","UNH","XOM",
        "JNJ","PG","MA","AVGO","HD","MRK","PEP","ABBV","COST","KO","PFE","ORCL","ADBE","CSCO","NFLX","TMO"
    ],
    "liquid100": [
        # Mega/Tech
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","AMD","CRM","ORCL","INTC","NFLX","MU","SMCI","PLTR",
        # Cloud/SaaS & Cyber
        "NOW","SNOW","DDOG","MDB","NET","PANW","CRWD","ZS","OKTA","SHOP","ADBE",
        # Semis/Hardware
        "QCOM","AMAT","LRCX","ASML","TSM","TXN","NXPI","ON",
        # Consumer/Disc/Staples
        "COST","WMT","HD","LOW","TGT","MCD","SBUX","NKE","LULU","EL","KO","PEP","PG","PM",
        # Autos & Mobility
        "GM","F","UBER","ABNB",
        # Financials/Payments
        "JPM","BAC","WFC","C","GS","MS","V","MA","AXP","PYPL",
        # Healthcare/Bio/MedTech
        "LLY","UNH","MRK","ABBV","JNJ","PFE","TMO","DHR","ISRG","MRNA","BMY","GILD","ZBH",
        # Energy/Materials
        "XOM","CVX","COP","SLB","EOG","PSX",
        # Industrials/Transports
        "CAT","BA","GE","DE","UPS","FDX","EMR","ETN",
        # Comm/Media/Other
        "DIS","CMCSA","SPOT","T","VZ"
    ],
    "etfs": ["SPY","QQQ","IWM"]
}

# Active universe (mix & match)
TICKERS = UNIVERSES["liquid100"] + UNIVERSES["etfs"]

# Primary (analysis) timeframe kept at 5m for indicator stability
INTERVAL = "5m"
PERIOD   = "5d"   # <= 60d for intraday
TIMEZONE = "America/Chicago"

# Prefilter scan timeframe (fast & cheap)
PREFILTER_INTERVAL = "1m"
PREFILTER_PERIOD   = "1d"

# RTH gating / market days
ENFORCE_RTH = True        # ← set False to allow scans anytime
MON_FRI_ONLY = True       # weekend guard
USE_US_HOLIDAYS = True   # requires `pip install holidays`; safe if left False
MARKET_CLOSE_BUFFER_MIN = 20  # don’t open new trades within this many minutes of close
_GATE_BLOCKED_REASON = None  # None | "not_market_day" | "after_hours" | "near_close"

# Prefilter debug knobs
PREFILTER_DEBUG = False
PREFILTER_DEBUG_PATH = "last_prefilter_debug.json"

STARTING_CAPITAL = 1000

# --- legacy % TP/SL (used as floors if ATR/delta conversion yields too small targets) ---
TP_PCT = 0.08     # 8% floor for TP (option mark)
SL_PCT = 0.05     # 5% floor for SL (option mark)

# --- budget & slippage ---
TOP_N = 5                    # how many candidates to consider/display
MAX_BUDGET_PER_TRADE = 1000  # hard cap for one contract, in USD
SLIPPAGE_BUY  = 0.005        # 0.5% buy slippage
SLIPPAGE_SELL = 0.005        # 0.5% sell slippage (used in PnL calc/logging)

# blending
TA_WEIGHT = 0.7
AI_WEIGHT = 0.3

# Indicators
ATR_WINDOW       = 14
ADX_WINDOW       = 14
MIN_ATR_PCT      = 0.5    # require at least 0.5% intraday vol (ATR/Close*100)
ADX_MIN_TREND    = 20.0   # trend-strength gate
BREAKOUT_PCT     = 0.002  # need close >= 0.2% above 20-bar high to count as breakout
VOL_SURGE_MULT   = 1.5    # volume confirmation multiplier vs 20-bar avg

# Selective scan thresholds / limits
TA_SCORE_MIN    = 1.6
TA_CONF_MIN     = 0.55
MAX_AI_EVALS    = 5
AI_CONF_MIN     = 0.45
AI_SCORE_MIN    = 1.5
AI_COOLDOWN_MIN = 1  # don’t re-AI-score same ticker within X minutes

# Contract selection rules ─────────────────────────────────────────────────
PREFER_3TO5_DTE_THU_FRI = True        # prefer 3–5 DTE when it's Thu/Fri
DELTA_TARGET_LO = 0.35                # target ~0.35–0.45 delta
DELTA_TARGET_HI = 0.45
TARGET_EXPECTED_RETURN_LO = 0.15      # aim 15–25% return for k×ATR favorable move
TARGET_EXPECTED_RETURN_HI = 0.25
ATR_K_LIST = [1.0, 1.5, 2.0]          # test 1×, 1.5×, 2× ATR moves
MAX_STRIKE_CANDIDATES = 20            # strikes to scan around ATM per expiry

OPEN_TRADES_JSON = "open_trades.json"
CLOSED_TRADES_CSV = "closed_trades.csv"
RECS_JSON = "last_recommendations.json"

# Logging files
LOG_DIR = "logs"
TA_LOG_CSV = os.path.join(LOG_DIR, "ta_candidates.csv")
AI_LOG_CSV = os.path.join(LOG_DIR, "ai_top3.csv")

# Robinhood (paper) – quotes only
RH_USERNAME = os.getenv("RH_USERNAME", "your_email@example.com")
RH_PASSWORD = os.getenv("RH_PASSWORD", "your_password")
DO_RH_LOGIN = True  # set True after credentials set

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

class TzFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tzinfo=None):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.tzinfo = tzinfo or tz.gettz(TIMEZONE)
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tzinfo)
        return dt.strftime(datefmt or "%Y-%m-%d %H:%M:%S %Z")
    
class TqdmHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            pass

logger = logging.getLogger("bot")
logger.setLevel(logging.INFO)
logger.propagate = False

_tqdm = TqdmHandler()
_tqdm.setLevel(logging.INFO)
_tqdm.setFormatter(TzFormatter("%(asctime)s %(levelname)s %(message)s"))

_stderr = logging.StreamHandler(stream=sys.stderr)
_stderr.setLevel(logging.ERROR)
_stderr.setFormatter(TzFormatter("%(asctime)s %(levelname)s %(message)s"))

if not logger.handlers:
    logger.addHandler(_tqdm)
    logger.addHandler(_stderr)

# ─────────────────────────────────────────────────────────────────────────────
# 1) UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def suppress_robinhood_noise():
    if os.getenv("RH_VERBOSE") == "1":
        yield
        return
    real_print = builtins.print
    try:
        builtins.print = lambda *args, **kwargs: None
        yield
    finally:
        builtins.print = real_print

def now_central():
    return datetime.now(tz.gettz(TIMEZONE))

def safe_mkdir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def read_open_trades():
    if not os.path.exists(OPEN_TRADES_JSON):
        return []
    with open(OPEN_TRADES_JSON, "r") as f:
        try:
            return json.load(f)
        except Exception:
            return []

def _as_float(x, default=0.0):
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def sum_open_cost(trades) -> float:
    """Cash tied up in currently open option(s), in dollars."""
    total = 0.0
    for t in trades:
        if t.get("status") == "OPEN":
            entry_basis = _as_float(t.get("entry_fill") or t.get("entry_mark"))
            total += entry_basis * 100.0
    return total

def compute_realized_pnl_from_csv() -> float:
    """Realized PnL from closed trades CSV, in dollars."""
    if not os.path.exists(CLOSED_TRADES_CSV):
        return 0.0
    total = 0.0
    with open(CLOSED_TRADES_CSV, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            entry_fill = _as_float(row.get("entry_fill") or row.get("entry_mark"))
            exit_fill  = _as_float(row.get("exit_fill")  or row.get("exit_mark"))
            total += (exit_fill - entry_fill) * 100.0
    return total

def compute_capital(open_trades_override=None) -> float:
    """Free cash = starting + realized PnL − cost of currently open positions."""
    current_opens = open_trades_override if open_trades_override is not None else read_open_trades()
    realized = compute_realized_pnl_from_csv()
    reserved = sum_open_cost(current_opens)
    return STARTING_CAPITAL + realized - reserved

def write_open_trades(trades):
    safe_mkdir(OPEN_TRADES_JSON)
    with open(OPEN_TRADES_JSON, "w") as f:
        json.dump(trades, f, indent=2, default=str)

def append_closed_trade(row_dict):
    safe_mkdir(CLOSED_TRADES_CSV)
    exists = os.path.exists(CLOSED_TRADES_CSV)
    with open(CLOSED_TRADES_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row_dict)

def write_recommendations_json(recs):
    with open(RECS_JSON, "w") as f:
        json.dump(recs, f, indent=2, default=str)

def rth_block_reason():
    """Return a short reason string if scans should be blocked, else None."""
    now = now_central()
    if not is_market_day(now):
        return "not_market_day"
    if not trading_window_open():
        return "after_hours"
    if near_close_buffer():
        return "near_close"
    return None

def append_csv_row(path, fieldnames, row):
    safe_mkdir(path)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)

# Market day / hours helpers

def is_market_day(dt):
    if MON_FRI_ONLY and dt.weekday() >= 5:
        return False
    if USE_US_HOLIDAYS:
        try:
            import holidays
            us_holidays = holidays.UnitedStates()
            if dt.date() in us_holidays:
                return False
        except Exception:
            pass
    return True

# Trading window & AI cooldown
_last_ai_eval = {}  # {ticker: datetime}

# ─────────────────────────────────────────────────────────────────────────────
# 2) TECHNICALS
# ─────────────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA, RSI, MACD, VWAP (session-reset), ATR(ewm), ADX(ewm)."""
    out = df.copy()

    # Flatten any (n,1) frames into Series
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in out.columns and isinstance(out[col], pd.DataFrame):
            out[col] = out[col].iloc[:, 0]
        else:
            out[col] = pd.Series(out[col].values, index=out.index)

    # EMA
    out["EMA9"]  = out["Close"].ewm(span=9,  adjust=False).mean()
    out["EMA21"] = out["Close"].ewm(span=21, adjust=False).mean()

    # RSI(14)
    delta = out["Close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=out.index).ewm(span=14, adjust=False).mean()
    roll_down = pd.Series(down, index=out.index).ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-9)
    out["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD
    ema12 = out["Close"].ewm(span=12, adjust=False).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    # VWAP (reset each session)
    tp = (out["High"] + out["Low"] + out["Close"]) / 3.0
    cum_vol = out["Volume"].groupby(out.index.date).cumsum().replace(0, np.nan)
    cum_tp_vol = (tp * out["Volume"]).groupby(out.index.date).cumsum()
    out["VWAP"] = cum_tp_vol / cum_vol

    # Rolling high/low & avg volume
    out["DayHigh20"] = out["High"].rolling(20).max()
    out["DayLow20"]  = out["Low"].rolling(20).min()
    out["VolMA20"]   = out["Volume"].rolling(20).mean()

    # ATR (EWMA of True Range)
    hl   = out["High"] - out["Low"]
    h_pc = (out["High"] - out["Close"].shift(1)).abs()
    l_pc = (out["Low"]  - out["Close"].shift(1)).abs()
    tr = pd.concat([hl, h_pc, l_pc], axis=1).max(axis=1)
    out["ATR"] = tr.ewm(span=ATR_WINDOW, adjust=False).mean()

    # ADX (EWMA variant)
    up_move   = out["High"].diff()
    down_move = -out["Low"].diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_ewm   = tr.ewm(span=ADX_WINDOW, adjust=False).mean()
    plus_di  = 100 * (pd.Series(plus_dm,  index=out.index).ewm(span=ADX_WINDOW, adjust=False).mean() / (tr_ewm + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=out.index).ewm(span=ADX_WINDOW, adjust=False).mean() / (tr_ewm + 1e-9))
    dx = 100 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di + 1e-9))
    out["ADX"] = dx.ewm(span=ADX_WINDOW, adjust=False).mean()

    return out


def evaluate_signal_ta(df: pd.DataFrame) -> dict:
    """Score the most recent bar using pure TA with safe scalar extraction."""
    if df is None or df.empty:
        return {"ta_score": 0.0, "ta_direction": "PUT", "ta_confidence": 0.0, "ta_reasons": ["no data"]}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    def val(row, name):
        v = row[name]
        try:
            return float(v) if pd.notna(v) else np.nan
        except Exception:
            return np.nan

    ema9, ema21 = val(last, "EMA9"), val(last, "EMA21")
    rsi = val(last, "RSI14")
    macd, macd_sig = val(last, "MACD"), val(last, "MACD_Signal")
    macd_prev, macd_sig_prev = val(prev, "MACD"), val(prev, "MACD_Signal")
    vwap = val(last, "VWAP")
    close_, prev_close = val(last, "Close"), val(prev, "Close")
    day_hi20, day_lo20 = val(last, "DayHigh20"), val(last, "DayLow20")
    vol, volma20 = val(last, "Volume"), val(last, "VolMA20")
    atr = val(last, "ATR")
    adx = val(last, "ADX")

    score = 0.0
    reasons = []

    # Volatility & trend-strength gates
    atr_pct = (atr / close_ * 100.0) if (pd.notna(atr) and pd.notna(close_) and close_ > 0) else np.nan
    vol_ok = pd.notna(atr_pct) and atr_pct >= MIN_ATR_PCT
    adx_ok = pd.notna(adx) and adx >= ADX_MIN_TREND

    if vol_ok: reasons.append(f"ATR% {atr_pct:.2f} ≥ {MIN_ATR_PCT:.2f} (vol OK)")
    else:      reasons.append(f"ATR% {atr_pct:.2f} < {MIN_ATR_PCT:.2f} (low vol; trend bonuses reduced)")
    if adx_ok: reasons.append(f"ADX {adx:.1f} ≥ {ADX_MIN_TREND:.0f} (trend OK)")
    else:      reasons.append(f"ADX {adx:.1f} < {ADX_MIN_TREND:.0f} (weak trend; trend bonuses reduced)")

    # EMA trend (weight by gates)
    ema_weight = 1.0 if (vol_ok and adx_ok) else 0.3
    if pd.notna(ema9) and pd.notna(ema21):
        if ema9 > ema21:
            score += 1.0 * ema_weight; reasons.append(f"EMA9>EMA21 (bullish){' [reduced]' if ema_weight<1 else ''}")
        elif ema9 < ema21:
            score -= 1.0 * ema_weight; reasons.append(f"EMA9<EMA21 (bearish){' [reduced]' if ema_weight<1 else ''}")

    # RSI
    if pd.notna(rsi):
        if 55 <= rsi < 65:
            score += 0.6; reasons.append("RSI 55–65 (bullish momentum)")
        elif 65 <= rsi < 70:
            score += 0.15; reasons.append("RSI 65–70 (momentum waning)")
        elif rsi >= 70:
            score -= 0.5; reasons.append("RSI ≥70 (overbought risk)")
        elif 45 <= rsi < 55:
            reasons.append("RSI 45–55 (neutral)")
        elif 30 <= rsi < 45:
            score -= 0.6; reasons.append("RSI 30–45 (bearish momentum)")
        elif rsi < 30:
            score += 0.3; reasons.append("RSI <30 (oversold risk/bounce)")

    # MACD
    if pd.notna(macd) and pd.notna(macd_sig) and pd.notna(macd_prev) and pd.notna(macd_sig_prev):
        if macd > macd_sig and macd_prev <= macd_sig_prev:
            score += 0.8; reasons.append("Fresh MACD bull cross")
        elif macd < macd_sig and macd_prev >= macd_sig_prev:
            score -= 0.8; reasons.append("Fresh MACD bear cross")
        else:
            score += 0.2 if macd > macd_sig else (-0.2 if macd < macd_sig else 0)

    # VWAP
    if pd.notna(close_) and pd.notna(vwap):
        if close_ > vwap: score += 0.3; reasons.append("Above VWAP")
        elif close_ < vwap: score -= 0.3; reasons.append("Below VWAP")

    # Breakout / breakdown with confirmation
    if pd.notna(close_) and pd.notna(day_hi20) and pd.notna(vol) and pd.notna(volma20) and volma20 > 0:
        if close_ >= day_hi20 * (1.0 + BREAKOUT_PCT) and vol > VOL_SURGE_MULT * volma20 and vol_ok and adx_ok:
            score += 0.6; reasons.append(f"Confirmed breakout (+{BREAKOUT_PCT*100:.1f}% & vol>{VOL_SURGE_MULT}×)")
        if pd.notna(day_lo20) and close_ <= day_lo20 * (1.0 - BREAKOUT_PCT) and vol > VOL_SURGE_MULT * volma20 and vol_ok and adx_ok:
            score -= 0.6; reasons.append(f"Confirmed breakdown (−{BREAKOUT_PCT*100:.1f}% & vol>{VOL_SURGE_MULT}×)")

    # Volume surge
    if pd.notna(vol) and pd.notna(volma20) and volma20 > 0:
        if vol > 1.5 * volma20 and pd.notna(prev_close):
            if close_ > prev_close: score += 0.4; reasons.append("Bullish volume surge")
            elif close_ < prev_close: score -= 0.4; reasons.append("Bearish volume surge")

    ta_direction = "CALL" if score > 0 else "PUT"
    ta_confidence = round(min(1.0, abs(score) / 3.2), 2)
    return {"ta_score": round(score, 2), "ta_direction": ta_direction, "ta_confidence": ta_confidence, "ta_reasons": reasons}

# ─────────────────────────────────────────────────────────────────────────────
# 3) AI OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

def ask_gpt_for_hybrid_score(ticker: str, df_ta: pd.DataFrame) -> dict:
    """Return {ai_direction, ai_confidence (0..1), rationale, risk_notes} or empty on failure."""
    if not USE_AI or _client is None:
        return {}
    last = df_ta.iloc[-1]
    prev = df_ta.iloc[-2] if len(df_ta) >= 2 else last
    context = {
        "ticker": ticker,
        "timestamp": df_ta.index[-1].isoformat(),
        "close": round(float(last["Close"]), 4),
        "ema9": round(float(last["EMA9"]), 4),
        "ema21": round(float(last["EMA21"]), 4),
        "rsi14": round(float(last["RSI14"]), 2),
        "macd": round(float(last["MACD"]), 4),
        "macd_signal": round(float(last["MACD_Signal"]), 4),
        "macd_hist": round(float(last["MACD_Hist"]), 4),
        "vwap": round(float(last["VWAP"]), 4),
        "day_high20": round(float(last["DayHigh20"]), 4),
        "day_low20": round(float(last["DayLow20"]), 4),
        "volume": int(last["Volume"]),
        "volma20": None if pd.isna(last["VolMA20"]) else int(last["VolMA20"]),
        "prev_close": round(float(prev["Close"]), 4),
    }
    system = (
        "You are a disciplined intraday trading analyst. "
        "Use only the structured indicator data to assess the next ~2 hours trend bias. "
        "No news or external context. Output strict JSON with fields: "
        "{ai_direction: 'CALL'|'PUT', ai_confidence: number 0..1, rationale: short string, risk_notes: short string}."
    )
    user = (
        "Given this 1–5 minute indicator snapshot (intraday), choose CALL if bias is up, PUT if down. "
        "Confidence should reflect strength and alignment of signals. Be conservative when signals conflict.\n\n"
        f"{json.dumps(context)}"
    )
    try:
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},{"role": "user", "content": user}],
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        ai_dir = str(data.get("ai_direction", "")).upper()
        if ai_dir not in ("CALL", "PUT"):
            return {}
        ai_conf = data.get("ai_confidence", None)
        try:
            ai_conf = float(ai_conf)
        except Exception:
            ai_conf = None
        if ai_conf is None or math.isnan(ai_conf):
            return {}
        ai_conf = max(0.0, min(1.0, ai_conf))
        return {
            "ai_direction": ai_dir,
            "ai_confidence": round(ai_conf, 2),
            "rationale": data.get("rationale", ""),
            "risk_notes": data.get("risk_notes", "")
        }
    except Exception:
        return {}


def blend_scores(ta_dir: str, ta_score: float, ta_conf: float, ai: dict) -> dict:
    """Blend TA and AI into final direction & score."""
    if not ai:
        return {"direction": ta_dir, "score": round(float(ta_score), 2), "confidence": round(float(ta_conf), 2), "blend_notes": "TA-only (AI unavailable)"}
    def sign(d): return 1 if d == "CALL" else -1
    ta_component = TA_WEIGHT * ta_score
    ai_component = AI_WEIGHT * (sign(ai["ai_direction"]) * 3.2 * ai["ai_confidence"])  # 3.2 ≈ TA scale
    blended = ta_component + ai_component
    direction = "CALL" if blended > 0 else "PUT"
    confidence = min(1.0, abs(blended) / 3.2)
    return {"direction": direction, "score": round(float(blended), 2), "confidence": round(float(confidence), 2), "blend_notes": f"blend TA({TA_WEIGHT}) + AI({AI_WEIGHT})"}

# ─────────────────────────────────────────────────────────────────────────────
# 4) DATA & PREFILTER / CONFIRM
# ─────────────────────────────────────────────────────────────────────────────

def _fix_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns and isinstance(df[col], pd.DataFrame):
            df[col] = df[col].iloc[:, 0]
    return df


def fetch_5m_df(ticker: str, latest_session_only: bool = True) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, progress=False, auto_adjust=False, prepost=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = _fix_df_columns(df)
    local_tz = tz.gettz(TIMEZONE)
    if df.index.tz is None:
        df = df.tz_localize("UTC").tz_convert(local_tz)
    else:
        df = df.tz_convert(local_tz)
    if latest_session_only:
        last_session = df.index[-1].date()
        df = df[df.index.date == last_session]
    return df


def fetch_1m_today(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PREFILTER_PERIOD, interval=PREFILTER_INTERVAL, progress=False, auto_adjust=False, prepost=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = _fix_df_columns(df)
    local_tz = tz.gettz(TIMEZONE)
    if df.index.tz is None:
        df = df.tz_localize("UTC").tz_convert(local_tz)
    else:
        df = df.tz_convert(local_tz)
    return df[df.index.date == now_central().date()]


def ta_prefilter(
    tickers,
    min_conf: float = 0.50,
    min_abs_score: float = 1.00,
    return_all: bool = False,
):
    diagnostics = []
    qualified = []

    for t in tickers:
        try:
            # Try true 1m today; fallback to 5m latest session if not enough bars
            df1 = fetch_1m_today(t)
            use_df = df1 if len(df1) >= 50 else fetch_5m_df(t, latest_session_only=True)
            if use_df is None or use_df.empty or len(use_df) < 50:  # require some history
                diagnostics.append({"ticker": t, "error": "no/insufficient bars", "pass": False})
                continue

            df_ta_full = compute_indicators(use_df)

            # Use the last available bar (today if present, otherwise last)
            today = now_central().date()
            df_today = df_ta_full[df_ta_full.index.date == today]
            row = df_today.iloc[-1] if len(df_today) else df_ta_full.iloc[-1]

            # Give evaluate_signal_ta a recent slice
            end_ts = row.name
            df_slice = df_ta_full.loc[:end_ts].tail(300)
            ta = evaluate_signal_ta(df_slice)

            def _num(x):
                try:
                    v = float(x)
                    return v if np.isfinite(v) else None
                except Exception:
                    return None

            close = _num(row.get("Close"))
            atr   = _num(row.get("ATR"))
            adx   = _num(row.get("ADX"))
            atr_pct = (atr / close * 100.0) if (atr is not None and close and close > 0) else None
            vol_ok = (atr_pct is not None and atr_pct >= MIN_ATR_PCT)
            adx_ok = (adx is not None and adx >= ADX_MIN_TREND)

            di = {
                "ticker": t,
                "ta_score": ta.get("ta_score"),
                "ta_confidence": ta.get("ta_confidence"),
                "ta_direction": ta.get("ta_direction"),
                "ta_reasons": ta.get("ta_reasons", []),
                "bars": int(len(df_slice)),
                "last": {
                    "close": close,
                    "ema9": _num(row.get("EMA9")),
                    "ema21": _num(row.get("EMA21")),
                    "rsi14": _num(row.get("RSI14")),
                    "macd": _num(row.get("MACD")),
                    "macd_signal": _num(row.get("MACD_Signal")),
                    "vwap": _num(row.get("VWAP")),
                    "atr": atr,
                    "adx": adx,
                    "atr_pct": (round(atr_pct, 3) if atr_pct is not None else None),
                    "vol_ok": vol_ok,
                    "adx_ok": adx_ok,
                    "volume": int(row.get("Volume") or 0),
                },
            }

            passed = (
                di["ta_score"] is not None and
                di["ta_confidence"] is not None and
                abs(float(di["ta_score"])) >= float(min_abs_score) and
                float(di["ta_confidence"]) >= float(min_conf)
            )
            di["pass"] = bool(passed)
            diagnostics.append(di)

            if passed:
                qualified.append({
                    "ticker": t,
                    "ta_score": di["ta_score"],
                    "ta_confidence": di["ta_confidence"],
                    "ta_direction": di["ta_direction"],
                    "ta": ta,
                    "df_ta": df_slice,  # for AI
                })

        except Exception as e:
            diagnostics.append({"ticker": t, "error": f"{e}", "pass": False})
            continue

    diagnostics_sorted = sorted(
        diagnostics,
        key=lambda x: (abs(x.get("ta_score") or 0.0), x.get("ta_confidence") or 0.0),
        reverse=True,
    )

    if PREFILTER_DEBUG:
        try:
            with open(PREFILTER_DEBUG_PATH, "w") as f:
                json.dump(diagnostics_sorted, f, indent=2, default=str)
        except Exception:
            pass

    return (qualified, diagnostics_sorted) if return_all else qualified


# Simple Robinhood expirations cache to reduce calls
_EXP_CACHE = {"data": {}, "ts": {}}  # symbol -> list[str]; ts for freshness
_EXP_CACHE_TTL_SEC = 1800  # 30 minutes


def _all_future_expirations(symbol: str) -> list[str]:
    """Cached full future expirations list (yyyy-mm-dd) for a symbol."""
    try:
        now_ts = time.time()
        cache_ok = symbol in _EXP_CACHE["data"] and (now_ts - _EXP_CACHE["ts"].get(symbol, 0)) < _EXP_CACHE_TTL_SEC
        if cache_ok:
            return _EXP_CACHE["data"][symbol]

        with suppress_robinhood_noise():
            all_opts = r.options.find_tradable_options(symbol) or []
        if not all_opts:
            return []
        today = now_central().date()
        exps = set()
        for o in all_opts:
            e = o.get("expiration_date")
            if not e:
                continue
            try:
                dt = dateparser.parse(e).date()
                if dt >= today:
                    exps.add(e)
            except Exception:
                continue
        ordered = sorted(exps, key=lambda s: dateparser.parse(s).date())
        _EXP_CACHE["data"][symbol] = ordered
        _EXP_CACHE["ts"][symbol] = now_ts
        return ordered
    except Exception:
        return []

def choose_sensible_expirations(symbol: str, max_to_try: int = 6) -> list[str]:
    """
    Prefer 3–5 DTE when it's Thu/Fri; else Friday-first ordering.
    Fallback to earliest future expiries.
    """
    exps = _all_future_expirations(symbol)
    if not exps:
        return []

    today = now_central().date()
    wkday = today.weekday()  # Mon=0 ... Sun=6
    if PREFER_3TO5_DTE_THU_FRI and wkday in (3, 4):  # Thu or Fri
        # Filter expiries with DTE in [3..5]
        dte_3to5 = []
        others = []
        for e in exps:
            dte = (dateparser.parse(e).date() - today).days
            (dte_3to5 if 3 <= dte <= 5 else others).append(e)
        if dte_3to5:
            return dte_3to5[:max_to_try]
        # else fall through to normal ordering below

    # Friday-first default
    fridays = [e for e in exps if dateparser.parse(e).date().weekday() == 4]
    others  = [e for e in exps if e not in fridays]
    ordered = fridays + others
    return ordered[:max_to_try]

def ai_confirm_candidates(cands, top_k: int = 5, min_ai_conf: float = 0.55):
    """
    Given prefiltered TA candidates (dicts that at least contain {"ticker"}),
    recompute TA here on 5m, run AI on the top_k by TA strength, and return the
    confirmed list sorted by blended confidence then abs(score).
    """
    if not USE_AI or _client is None:
        return []  # AI disabled → no confirmations

    if not cands:
        return []

    # Rank by TA strength if available; otherwise keep input order
    def _ta_key(c):
        return (abs(c.get("ta_score", 0.0)), c.get("ta_confidence", 0.0))

    short_list = sorted(cands, key=_ta_key, reverse=True)[:top_k]

    confirmed = []
    evals = 0

    for c in short_list:
        tkr = c.get("ticker")
        if not tkr:
            continue

        # AI cooldown per ticker
        ts_last = _last_ai_eval.get(tkr)
        if ts_last and (now_central() - ts_last).total_seconds() / 60.0 < AI_COOLDOWN_MIN:
            continue

        try:
            df = fetch_5m_df(tkr)
            if df is None or df.empty or len(df) < 26:
                continue

            df_ta = compute_indicators(df)
            ta = evaluate_signal_ta(df_ta)

            ai = ask_gpt_for_hybrid_score(tkr, df_ta)
            if not ai or ai.get("ai_confidence", 0.0) < float(min_ai_conf):
                continue

            _last_ai_eval[tkr] = now_central()
            evals += 1

            blended = blend_scores(
                ta_dir=ta["ta_direction"],
                ta_score=ta["ta_score"],
                ta_conf=ta["ta_confidence"],
                ai=ai
            )

            last = df_ta.iloc[-1]
            entry_price = float(last["Close"])

            # ATR-based underlying TP/SL for context
            try:
                atr_val = float(last.get("ATR"))
            except Exception:
                atr_val = float("nan")
            if not (np.isfinite(atr_val) and atr_val > 0):
                atr_val = max(0.005 * entry_price, 0.25)

            # --- NEW: ADX-based TP multiplier ---
            try:
                adx_val = float(last.get("ADX"))
            except Exception:
                adx_val = float("nan")

            # Fallback rr if ADX is missing/invalid
            default_rr = 1.5

            if not (np.isfinite(adx_val) and adx_val > 0):
                rr_mult = default_rr
            elif adx_val < 25:
                rr_mult = 1.2       # weak/sideways → tighter TP
            elif adx_val < 35:
                rr_mult = 1.5       # moderate trend
            else:
                rr_mult = 2.0       # strong trend → stretch TP a bit

            # Compute TP/SL on the underlying
            if blended["direction"] == "CALL":
                tp_under = round(entry_price + rr_mult * atr_val, 2)
                sl_under = round(entry_price - 1.0 * atr_val, 2)
            else:
                tp_under = round(entry_price - rr_mult * atr_val, 2)
                sl_under = round(entry_price + 1.0 * atr_val, 2)

            # Minimal indicators payload for downstream logging/opening
            indicators = {
                "ATR": float(last.get("ATR")),
                "ADX": float(last.get("ADX")),
                "RSI14": float(last.get("RSI14")),
                "EMA9": float(last.get("EMA9")),
                "EMA21": float(last.get("EMA21")),
                "VWAP": float(last.get("VWAP")),
            }

            confirmed.append({
                "ticker": tkr,
                "timestamp": df_ta.index[-1].isoformat(),
                "direction": blended["direction"],
                "score": blended["score"],
                "confidence": blended["confidence"],
                "entry_price": entry_price,
                "tp_underlying": tp_under,
                "sl_underlying": sl_under,
                "blend_notes": blended["blend_notes"],
                "indicators": indicators,
                "ta": ta,
                "ai": ai,
            })

            if evals >= MAX_AI_EVALS:
                break

        except Exception as e:
            continue

    return sorted(
        confirmed,
        key=lambda x: (x["confidence"], abs(x["score"])),
        reverse=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# 5) ROBINHOOD PAPER QUOTE HELPERS (OPTIONS)
# ─────────────────────────────────────────────────────────────────────────────

def rh_login():
    try:
        r.authentication.logout()
    except Exception:
        pass
    return r.authentication.login(username=RH_USERNAME, password=RH_PASSWORD, expiresIn=86400, by_sms=True)


def get_option_delta(opt_id: str) -> float | None:
    """Try to read option delta from Robinhood; return absolute delta in [0,1] or None."""
    try:
        with suppress_robinhood_noise():
            md = r.options.get_option_market_data_by_id(opt_id)
        if isinstance(md, list):
            md = md[0] if md else {}
        raw = md.get("delta")
        if raw in (None, "", "None"):
            return None
        d = abs(float(raw))
        if 0.01 <= d <= 1.0:
            return d
        return None
    except Exception:
        return None


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def choose_weekly_expiration(symbol: str, max_to_try: int = 4) -> list[str]:
    """Prefer future Fridays, then others. Cached for ~30 minutes."""
    try:
        now_ts = time.time()
        cache_ok = symbol in _EXP_CACHE["data"] and (now_ts - _EXP_CACHE["ts"].get(symbol, 0)) < _EXP_CACHE_TTL_SEC
        if cache_ok:
            return _EXP_CACHE["data"][symbol][:max_to_try]

        with suppress_robinhood_noise():
            all_opts = r.options.find_tradable_options(symbol) or []
        if not all_opts:
            return []
        today = date.today()
        exps = set()
        for o in all_opts:
            e = o.get("expiration_date")
            if not e:
                continue
            try:
                dt = dateparser.parse(e).date()
                if dt >= today:
                    exps.add(e)
            except Exception:
                continue
        if not exps:
            return []
        exps_sorted = sorted(exps, key=lambda s: dateparser.parse(s).date())
        fridays = [e for e in exps_sorted if dateparser.parse(e).date().weekday() == 4]
        others  = [e for e in exps_sorted if e not in fridays]
        ordered = fridays + others

        _EXP_CACHE["data"][symbol] = ordered
        _EXP_CACHE["ts"][symbol] = now_ts
        return ordered[:max_to_try]
    except Exception:
        return []

def get_option_md(opt_id: str) -> dict:
    """Single-call MD: returns {'mark','bid','ask','delta'} (floats or None)."""
    try:
        with suppress_robinhood_noise():
            md = r.options.get_option_market_data_by_id(opt_id)
        if isinstance(md, list):
            md = md[0] if md else {}
        if not md:
            return {"mark": None, "bid": None, "ask": None, "delta": None}
        def fnum(k):
            v = md.get(k)
            try:
                x = float(v)
                return x if math.isfinite(x) else None
            except Exception:
                return None
        out = {
            "mark": fnum("mark_price") or (lambda b,a: (b+a)/2 if (b is not None and a is not None) else None)(fnum("bid_price"), fnum("ask_price")) or fnum("last_trade_price") or fnum("adjusted_mark_price"),
            "bid":  fnum("bid_price"),
            "ask":  fnum("ask_price"),
            "delta": (lambda d: abs(d) if d is not None and 0.01 <= abs(d) <= 1.0 else None)(fnum("delta")),
        }
        return out
    except Exception:
        return {"mark": None, "bid": None, "ask": None, "delta": None}

def _option_chain_for(symbol: str, expiration: str, opt_type: str):
    """Return list of option instrument dicts for a symbol/expiry/type."""
    try:
        with suppress_robinhood_noise():
            all_opts = r.options.find_tradable_options(symbol) or []
        return [o for o in all_opts if o.get("expiration_date") == expiration and o.get("type") == opt_type]
    except Exception:
        return []


def pick_option_contract(symbol: str, direction: str, underlying_price: float, expirations: list[str], atr: float | None = None) -> tuple[dict, str | None]:
    """
    Priority:
      1) If ATR provided: pick contracts where expected return (delta * k*ATR / price) ∈ [15%,25%] for some k in ATR_K_LIST,
         nearest to 20% and with smaller spread%.
      2) Else: pick ~0.35–0.45 delta nearest to 0.40 (then smaller spread%).
      3) Fallback: ATM (old method).
    Returns ({option_symbol, option_id, strike, expiration, type, initial_mark}, reason|None)
    """
    opt_type = "call" if direction == "CALL" else "put"
    best = None
    best_key = None
    reason = None

    def _spread_pct(bid, ask, mark):
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return float("inf")
        spr = max(0.0, ask - bid)
        base = mark if (mark and mark > 0) else (0.5*(bid+ask))
        return spr / base if base and base > 0 else float("inf")

    # Scan expirations
    for exp in expirations:
        chain = _option_chain_for(symbol, exp, opt_type)
        if not chain:
            continue

        # Take ~ATM ± N strikes
        def _strike(o):
            try: return float(o.get("strike_price"))
            except: return float("inf")
        near_atm = sorted(chain, key=lambda o: abs(_strike(o) - underlying_price))[:MAX_STRIKE_CANDIDATES]

        # Pass 1: ATR targeting if ATR given
        if atr and atr > 0:
            for inst in near_atm:
                opt_id = inst.get("id")
                strike_val = _strike(inst)
                if not opt_id or not math.isfinite(strike_val):
                    continue
                md = get_option_md(opt_id)
                mark, bid, ask, delta = md["mark"], md["bid"], md["ask"], md["delta"]
                if mark is None or mark <= 0 or delta is None:
                    continue
                # expected return for k×ATR move
                exp_rets = [delta * (k * atr) / mark for k in ATR_K_LIST]
                # choose closest in-range; else skip to next pass
                in_range = [er for er in exp_rets if TARGET_EXPECTED_RETURN_LO <= er <= TARGET_EXPECTED_RETURN_HI]
                if not in_range:
                    continue
                # select by distance to midpoint + spread
                dist = min(abs(er - 0.5*(TARGET_EXPECTED_RETURN_LO + TARGET_EXPECTED_RETURN_HI)) for er in in_range)
                spct = _spread_pct(bid, ask, mark)
                key = (dist, spct, abs(delta - 0.40))
                pick = {
                    "option_symbol": inst.get("symbol") or inst.get("id"),
                    "option_id": opt_id,
                    "strike": float(strike_val),
                    "expiration": exp,
                    "type": opt_type,
                    "initial_mark": float(mark)
                }
                if best is None or key < best_key:
                    best, best_key, reason = pick, key, f"ATR-targeted (k∈{ATR_K_LIST})"
            if best:
                return best, reason  # stop after first expiry with a good ATR-targeted hit

        # Pass 2: Delta window 0.35–0.45
        delta_candidates = []
        for inst in near_atm:
            opt_id = inst.get("id")
            strike_val = _strike(inst)
            if not opt_id or not math.isfinite(strike_val):
                continue
            md = get_option_md(opt_id)
            mark, bid, ask, delta = md["mark"], md["bid"], md["ask"], md["delta"]
            if mark is None or mark <= 0 or delta is None:
                continue
            if DELTA_TARGET_LO <= delta <= DELTA_TARGET_HI:
                delta_candidates.append((abs(delta - 0.40), _spread_pct(bid, ask, mark), {
                    "option_symbol": inst.get("symbol") or inst.get("id"),
                    "option_id": opt_id,
                    "strike": float(strike_val),
                    "expiration": exp,
                    "type": opt_type,
                    "initial_mark": float(mark)
                }))
        if delta_candidates:
            delta_candidates.sort(key=lambda t: (t[0], t[1]))
            pick = delta_candidates[0][2]
            return pick, "Delta-targeted 0.35–0.45"

    # Pass 3: Fallback to ATM
    fallback, err = pick_atm_option_symbol(symbol, direction, underlying_price, expirations)
    return (fallback or {}), (err or "Fallback ATM")

def pick_atm_option_symbol(symbol: str, direction: str, underlying_price: float, expirations: list[str]) -> tuple[dict, str | None]:
    """Choose nearest-to-ATM contract with usable market data."""
    opt_type = "call" if direction == "CALL" else "put"
    last_err = "No expirations provided."
    try:
        with suppress_robinhood_noise():
            all_opts = r.options.find_tradable_options(symbol) or []
        if not all_opts:
            return {}, f"No tradable options for {symbol}."
        for exp in expirations or []:
            chain = [o for o in all_opts if o.get("expiration_date") == exp and o.get("type") == opt_type]
            if not chain:
                last_err = f"No {opt_type} chain for {symbol} {exp}"
                continue
            def _strike(o):
                try: return float(o.get("strike_price"))
                except Exception: return float("inf")
            candidates = sorted(chain, key=lambda o: abs(_strike(o) - underlying_price))
            for inst in candidates[:10]:
                opt_id = inst.get("id")
                strike_val = _strike(inst)
                if not opt_id or not math.isfinite(strike_val):
                    continue
                mark = get_option_mark_by_id(opt_id)
                if mark is None:
                    continue
                return {
                    "option_symbol": inst.get("symbol") or inst.get("id"),
                    "option_id": opt_id,
                    "strike": float(strike_val),
                    "expiration": exp,
                    "type": opt_type,
                    "initial_mark": float(mark)
                }, None
            last_err = f"No usable quotes near ATM for {symbol} {opt_type} {exp}"
        return {}, last_err
    except Exception as e:
        return {}, f"Chain selection error for {symbol} {opt_type}: {e}"


def get_option_mark_by_id(opt_id: str) -> float | None:
    """Fetch market data by instrument ID and compute a usable mark."""
    try:
        with suppress_robinhood_noise():
            md = r.options.get_option_market_data_by_id(opt_id)
        if isinstance(md, list):
            md = md[0] if md else {}
        if not isinstance(md, dict) or not md:
            return None
        def f(k):
            v = md.get(k)
            try:
                return float(v) if v not in (None, "", "None") else None
            except Exception:
                return None
        mark = f("mark_price")
        if mark is not None:
            return mark
        bid, ask = f("bid_price"), f("ask_price")
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        last = f("last_trade_price")
        if last is not None:
            return last
        adj = f("adjusted_mark_price")
        return adj
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# 6) TRADE STATE
# ─────────────────────────────────────────────────────────────────────────────

def open_new_trade_from_reco(reco):
    """Open 1 long option based on reco; TP/SL in option-price terms via ATR/delta conversion."""
    ticker = reco["ticker"]
    direction = reco["direction"]
    underlying_entry = reco["entry_underlying"]

    ind = (reco.get("setup", {}) or {}).get("indicators", {}) if reco.get("setup") else {}
    atr = ind.get("ATR", reco.get("atr"))
    try:
        atr = float(atr) if atr is not None else float("nan")
    except Exception:
        atr = float("nan")
    if not (np.isfinite(atr) and atr > 0):
        atr = None

    expiries = choose_sensible_expirations(ticker, max_to_try=6)
    if not expiries:
        return None, f"No expirations available for {ticker}."

    # Pick contract (ATR-targeted → delta-targeted → ATM fallback)
    opt, err = pick_option_contract(ticker, direction, underlying_entry, expiries, atr=atr)
    if not opt:
        return None, err or "Could not resolve option contract."

    selection_reason = err or "ATM"

    option_symbol = opt.get("option_symbol")
    option_id     = opt.get("option_id")
    entry_mark    = opt.get("initial_mark")

    if entry_mark is None:
        entry_mark = get_option_mark_by_id(option_id)
    if entry_mark is None:
        return None, (f"Could not fetch mark for {ticker} {direction} {opt.get('expiration')} {opt.get('strike')} "
                      f"(id={option_id}, sym={option_symbol})")

    entry_fill = float(entry_mark) * (1.0 + SLIPPAGE_BUY)
    est_cost   = entry_fill * 100.0
    if est_cost > MAX_BUDGET_PER_TRADE:
        return None, f"Estimated cost ${est_cost:.2f} exceeds budget ${MAX_BUDGET_PER_TRADE:.2f}"

    # Convert underlying ATR targets → option TP/SL via delta (fallback 0.50)
    und_entry = float(reco.get("entry_underlying", underlying_entry))
    und_tp    = float(reco.get("tp_underlying", und_entry))
    und_sl    = float(reco.get("sl_underlying", und_entry))

    if direction == "CALL":
        favorable_move = max(0.0, und_tp - und_entry)
        adverse_move   = max(0.0, und_entry - und_sl)
    else:
        favorable_move = max(0.0, und_entry - und_tp)
        adverse_move   = max(0.0, und_sl - und_entry)

    delta_est = get_option_delta(option_id)
    if delta_est is None:
        logger.info(f"[open] {ticker} missing delta, fallback to 0.50")
        delta_est = 0.50
    delta_est = _clamp(delta_est, 0.10, 0.85)

    opt_fav_change = delta_est * favorable_move
    opt_adv_change = delta_est * adverse_move

    # floors to avoid tiny targets on quiet days
    min_tp_fallback = entry_fill * TP_PCT
    min_sl_fallback = entry_fill * SL_PCT
    if opt_fav_change < min_tp_fallback: opt_fav_change = min_tp_fallback
    if opt_adv_change < min_sl_fallback: opt_adv_change = min_sl_fallback

    tp_mark = entry_fill + opt_fav_change
    sl_mark = entry_fill - opt_adv_change

    tp_mark = _clamp(tp_mark, 0.01, entry_fill * 10.0)
    sl_mark = _clamp(sl_mark, 0.01, entry_fill * 1.0)  # never above entry

    trade = {
        "id": f"{ticker}-{int(time.time())}",
        "opened_at": now_central().isoformat(),
        "ticker": ticker,
        "direction": direction,
        "option_symbol": option_symbol,
        "option_id": option_id,
        "expiration": opt.get("expiration"),
        "strike": opt.get("strike"),
        "side": "LONG",
        "entry_mark": round(float(entry_mark), 4),
        "entry_fill": round(float(entry_fill), 4),
        "tp_mark": round(float(tp_mark), 4),
        "sl_mark": round(float(sl_mark), 4),
        "atr": atr,
        "rr_tp_mult": 2.5,
        "rr_sl_mult": 1.0,
        "status": "OPEN",
        "notes": (
            f"{selection_reason}; paper via quotes only; "
            f"blend={TA_WEIGHT}/{AI_WEIGHT if USE_AI else 0}; "
            f"slippage buy/sell={SLIPPAGE_BUY:.2%}/{SLIPPAGE_SELL:.2%}"
        ),
        "last_check": now_central().isoformat(),
        "delta_est": round(delta_est, 3),
        "und_entry": und_entry,
        "und_tp": und_tp,
        "und_sl": und_sl,
    }

    existing = read_open_trades()
    capital_after_open = STARTING_CAPITAL + compute_realized_pnl_from_csv() - (sum_open_cost(existing) + (entry_fill * 100.0))
    trade["capital_available"] = round(capital_after_open, 2)

    return trade, None


def monitor_open_trades():
    trades = read_open_trades()
    keep = []
    closed_rows = []   # batch write after loop
    any_change = False

    # Keep a rolling cash snapshot for anything still open while we loop
    current_cash = compute_capital(trades)
    for t in trades:
        if t.get("status") != "OPEN":
            continue

        # always carry a live cash snapshot in file
        t["capital_available"] = round(current_cash, 2)

        mark = get_option_mark_by_id(t.get("option_id"))
        t["last_check"] = now_central().isoformat()
        if mark is None:
            keep.append(t)
            continue

        mark = float(mark)
        t["last_mark"] = round(mark, 4)

        entry_basis = float(t.get("entry_fill") or t["entry_mark"])
        sim_exit = mark * (1.0 - SLIPPAGE_SELL)

        pnl_dollars = (sim_exit - entry_basis) * 100.0
        roi_pct = ((pnl_dollars / (entry_basis * 100.0)) * 100.0) if entry_basis > 0 else 0.0

        logger.info(
            f"[monitor] {t['ticker']} {t['direction']} "
            f"entry={entry_basis:.3f} mark={mark:.3f} "
            f"sim_exit={sim_exit:.3f} PnL=${pnl_dollars:.2f} ROI={roi_pct:.2f}% "
            f"TP={t.get('tp_mark')} SL={t.get('sl_mark')}"
        )

        # 1R trail to breakeven
        sl_price = float(t.get("sl_mark"))
        risk_price = max(0.0, entry_basis - sl_price)
        risk_1R_opt = risk_price * 100.0
        if pnl_dollars >= risk_1R_opt and risk_1R_opt > 0:
            old_sl = float(t["sl_mark"])
            be_stop = entry_basis
            if be_stop > old_sl + 1e-6:
                t["sl_mark"] = round(be_stop, 4)
                t["trail_armed"] = True
                logger.info(f"[trail] {t['ticker']} +1R reached. Move SL → breakeven (old={old_sl:.3f}, new={t['sl_mark']:.3f})")
                any_change = True

            peak = float(t.get("peak_mark") or entry_basis)
            if mark > peak:
                t["peak_mark"] = round(mark, 4)

        # TP / SL checks
        hit_tp = mark >= float(t["tp_mark"])
        hit_sl = mark <= float(t["sl_mark"])

        if hit_tp or hit_sl:
            t["status"] = "CLOSED_TP" if hit_tp else "CLOSED_SL"
            t["closed_at"] = now_central().isoformat()
            exit_fill = round(mark * (1.0 - SLIPPAGE_SELL), 4)
            t["exit_mark"] = exit_fill  # kept original field name
            pnl_pct = round((exit_fill - entry_basis) / entry_basis, 4)

            # collect closed row; we'll add capital_after later in one shot
            closed_rows.append({
                "id": t["id"],
                "ticker": t["ticker"],
                "direction": t["direction"],
                "option_symbol": t.get("option_symbol"),
                "opened_at": t["opened_at"],
                "closed_at": t["closed_at"],
                "entry_mark": t["entry_mark"],
                "entry_fill": t["entry_fill"],   # new: actual cost basis
                "exit_mark": exit_fill,
                "exit_fill": exit_fill,          # new: actual proceeds
                "pnl_pct": pnl_pct,
                "reason": "TP" if hit_tp else "SL"
            })
            any_change = True
            # do NOT append to keep (it’s closed)
        else:
            keep.append(t)

    # After loop: compute capital AFTER applying all these closures
    if any_change:
        # realized PnL so far in CSV + new closures from this pass
        realized_so_far = compute_realized_pnl_from_csv()
        new_realized = 0.0
        for row in closed_rows:
            entry_fill = _as_float(row["entry_fill"] or row["entry_mark"])
            exit_fill  = _as_float(row["exit_fill"]  or row["exit_mark"])
            new_realized += (exit_fill - entry_fill) * 100.0

        capital_after = STARTING_CAPITAL + realized_so_far + new_realized - sum_open_cost(keep)

        # stamp capital snapshot on all remaining opens
        cap_rounded = round(capital_after, 2)
        for ot in keep:
            if ot.get("status") == "OPEN":
                ot["capital_available"] = cap_rounded

        # write open file first (closed positions removed)
        write_open_trades(keep)

        # append closed rows with a cash snapshot and emit one-line close logs
        for row in closed_rows:
            row["capital_after"] = cap_rounded  # new column in CSV
            append_closed_trade(row)

            # pretty one-liner in console
            sym = row["ticker"]
            side = row["direction"]
            efill = _as_float(row["entry_fill"] or row["entry_mark"])
            xfill = _as_float(row["exit_fill"]  or row["exit_mark"])
            pnl_usd = (xfill - efill) * 100.0
            pnl_pc = ((xfill - efill) / efill * 100.0) if efill > 0 else 0.0
            logger.info(
                f"[close] {sym} {side} {row['reason']} "
                f"entry={efill:.3f} exit={xfill:.3f} "
                f"PnL=${pnl_usd:.2f} ({pnl_pc:.2f}%) cash=${cap_rounded:.2f}"
            )
    else:
        # even if nothing closed, keep a fresh cash snapshot on opens
        cap_rounded = round(compute_capital(keep or trades), 2)
        touched = False
        for ot in keep or trades:
            if ot.get("status") == "OPEN":
                if ot.get("capital_available") != cap_rounded:
                    ot["capital_available"] = cap_rounded
                    touched = True
        if touched:
            write_open_trades(keep or trades)


def maybe_open_top_trade(recs_json):
    trades = read_open_trades()
    if any(t.get("status") == "OPEN" for t in trades):
        logger.info("[open] A trade is already OPEN; skipping new entry.")
        return

    existing = {(t["ticker"], t["direction"]) for t in trades if t.get("status") == "OPEN"}
    candidates = (recs_json.get("top_trades", []) or [])[:TOP_N]
    if not candidates:
        logger.info("[open] No candidates.")
        return

    for cand in candidates:
        key = (cand["ticker"], cand["direction"])
        if key in existing:
            logger.info(f"[open] Skipping {cand['ticker']} {cand['direction']}: already open.")
            continue
        trade, err = open_new_trade_from_reco(cand)
        if err or not trade:
            logger.info(f"[open] {cand['ticker']} {cand['direction']} not taken: {err}")
            continue
        trades.append(trade)
        write_open_trades(trades)
        logger.info(
            f"[open] NEW {trade['ticker']} {trade['direction']} "
            f"fill={trade['entry_fill']} (mark={trade['entry_mark']}) → "
            f"TP {trade['tp_mark']} / SL {trade['sl_mark']} "
            f"(budget ≤ ${MAX_BUDGET_PER_TRADE:.2f}) cash=${trade.get('capital_available', 0):.2f}"
        )
        return  # take only one
    logger.info("[open] No affordable candidates in the top list.")

# ─────────────────────────────────────────────────────────────────────────────
# 7) LOGGING HELPERS (TA candidates & AI Top 3)
# ─────────────────────────────────────────────────────────────────────────────

def log_ta_candidates(ta_diag, top_preview: int = 10):
    """Append all TA-passed candidates to CSV and print a concise preview."""
    as_of = now_central().isoformat()
    passed = [d for d in ta_diag if d.get("pass")]
    if not passed:
        # logger.info("[TA] No candidates passed thresholds.")
        return

    # CSV append
    fields = ["as_of","ticker","ta_score","ta_confidence","ta_direction","close","atr","atr_pct","adx","bars"]
    for d in passed:
        last = d.get("last", {})
        row = {
            "as_of": as_of,
            "ticker": d.get("ticker"),
            "ta_score": d.get("ta_score"),
            "ta_confidence": d.get("ta_confidence"),
            "ta_direction": d.get("ta_direction"),
            "close": last.get("close"),
            "atr": last.get("atr"),
            "atr_pct": last.get("atr_pct"),
            "adx": last.get("adx"),
            "bars": d.get("bars"),
        }
        append_csv_row(TA_LOG_CSV, fields, row)

    # Console preview
    ordered = sorted(
    passed,
    key=lambda x: (abs(x.get("ta_score") or 0.0), x.get("ta_confidence") or 0.0),
    reverse=True,
    )
    lines = []
    for i, d in enumerate(ordered[:top_preview], start=1):
        last = d.get("last", {})
        lines.append(
            f"#{i:>2} {d['ticker']:>5} | sc={d.get('ta_score'):>4} "
            f"cf={d.get('ta_confidence'):>4} dir={d.get('ta_direction','?'):>4} "
            f"ATR%={last.get('atr_pct')} ADX={last.get('adx')}"
        )
    logger.info("[TA] Passed candidates (top preview):\n" + "\n".join("  - " + s for s in lines))


def log_ai_top3(confirmed):
    """Append Top 3 AI-confirmed to CSV and print to console."""
    if not confirmed:
        # logger.info("[AI] No AI-confirmed candidates.")
        return
    as_of = now_central().isoformat()
    top3 = confirmed[:3]
    fields = ["as_of","rank","ticker","direction","confidence","score","entry_price","tp_underlying","sl_underlying","ai_confidence","ai_direction"]
    for i, rj in enumerate(top3, start=1):
        ai = rj.get("ai", {})
        row = {
            "as_of": as_of,
            "rank": i,
            "ticker": rj.get("ticker"),
            "direction": rj.get("direction"),
            "confidence": rj.get("confidence"),
            "score": rj.get("score"),
            "entry_price": rj.get("entry_price"),
            "tp_underlying": rj.get("tp_underlying"),
            "sl_underlying": rj.get("sl_underlying"),
            "ai_confidence": ai.get("ai_confidence"),
            "ai_direction": ai.get("ai_direction"),
        }
        append_csv_row(AI_LOG_CSV, fields, row)

    # Console preview
    lines = [
        f"#{i} {rj['ticker']} {rj['direction']} conf={rj['confidence']:.2f} "
        f"score={rj['score']:.2f} entry={rj['entry_price']:.2f}"
        for i, rj in enumerate(top3, start=1)
    ]
    indented = ["  - " + s for s in lines]
    logger.info("[AI] Top 3 confirmed:\n" + "\n".join(indented))

# ─────────────────────────────────────────────────────────────────────────────
# 8) SELECTIVE CYCLE & MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def trading_window_open() -> bool:
    now = now_central()
    if not is_market_day(now):
        return False
    # US equities RTH in CT: 08:30–15:00
    start = now.replace(hour=8, minute=30, second=0, microsecond=0)
    end   = now.replace(hour=15, minute=0, second=0, microsecond=0)
    return start <= now <= end


def near_close_buffer() -> bool:
    now = now_central()
    close = now.replace(hour=15, minute=0, second=0, microsecond=0)  # 3:00 PM CT
    return (close - now).total_seconds() / 60.0 <= MARKET_CLOSE_BUFFER_MIN


def has_open_trade() -> bool:
    return any(t.get("status") == "OPEN" for t in read_open_trades())


def ensure_rh():
    if not DO_RH_LOGIN:
        return
    try:
        r.profiles.load_account_profile()
    except Exception:
        logger.info("[rh] Logging in…")
        rh_login()


def run_cycle_selective():
    ensure_rh()
    # 1) Manage exits first
    monitor_open_trades()

    # 2) If any open position, do NOT scan
    if has_open_trade():
        # logger.info("[cycle] Position OPEN → monitoring only; no scanning.")
        return

    # 3) Scan only during RTH unless explicitly allowed
    if ENFORCE_RTH:
        global _GATE_BLOCKED_REASON
        reason = rth_block_reason()
        if reason is not None:
            if _GATE_BLOCKED_REASON != reason:
                # prints once when entering a blocked state (per reason)
                pretty = {
                    "not_market_day": "Non-market day",
                    "after_hours": "Outside trading window",
                    "near_close": "Within close buffer",
                }.get(reason, reason)
                logger.info(f"[cycle] {pretty} → skipping scan.")
                _GATE_BLOCKED_REASON = reason
            return
        else:
            # cleared: allow future one-time logs if/when we block again
            if _GATE_BLOCKED_REASON is not None:
                _GATE_BLOCKED_REASON = None

    # 4) Cheap TA prefilter on 1m (with full diagnostics)
    ta_cands, ta_diag = ta_prefilter(
        TICKERS,
        min_conf=TA_CONF_MIN,
        min_abs_score=TA_SCORE_MIN,
        return_all=True,
    )

    # Log TA potential trades
    log_ta_candidates(ta_diag)

    if not ta_cands:
        logger.info("[cycle] No TA candidates passed thresholds → flat.")
        return

    # 5) AI-confirm only the best few
    confirmed = ai_confirm_candidates(
        ta_cands,
        top_k=min(len(ta_cands), MAX_AI_EVALS),
        min_ai_conf=AI_CONF_MIN,   # ← add this
    )

    # Log AI Top 3
    log_ai_top3(confirmed)

    if not confirmed:
        logger.info("[cycle] No AI-confirmed candidates (cooldown/thresholds) → flat.")
        return

    # 6) Keep only strong AI ideas
    strong = [rj for rj in confirmed if rj["confidence"] >= AI_CONF_MIN and rj["score"] >= AI_SCORE_MIN]
    if not strong:
        logger.info("[cycle] AI ideas below confidence/score gates → flat.")
        return

    # 7) Form JSON compatible with maybe_open_top_trade
    out_json = {
        "as_of": now_central().isoformat(),
        "universe": TICKERS,
        "strategy": {
            "type": "hybrid_ta_ai" if USE_AI else "ta_only",
            "ta_weight": TA_WEIGHT,
            "ai_weight": AI_WEIGHT if USE_AI else 0.0,
            "model": OPENAI_MODEL if USE_AI else None
        },
        "top_trades": [
            {
                "rank": i + 1,
                "ticker": rj["ticker"],
                "direction": rj["direction"],
                "confidence": rj["confidence"],
                "score": rj["score"],
                "timestamp": rj["timestamp"],
                "entry_underlying": rj["entry_price"],
                "tp_underlying": rj["tp_underlying"],
                "sl_underlying": rj["sl_underlying"],
                "blend_notes": rj["blend_notes"],
                "setup": {
                    "indicators": rj["indicators"],
                    "ta": rj["ta"],
                    "ai": rj["ai"]
                }
            } for i, rj in enumerate(strong[:TOP_N])
        ]
    }
    write_recommendations_json(out_json)

    # 8) Attempt single open (affordability checked inside)
    maybe_open_top_trade(out_json)


def main_loop():
    logger.info("Options Bot starting (V1.0.3)…")
    ensure_rh()

    # Ensure state files exist
    if not os.path.exists(OPEN_TRADES_JSON):
        write_open_trades([])

    schedule.every(2).minutes.do(run_cycle_selective)
    logger.info("Scheduler started: selective cycle every 2 minutes.")

    # Kick off immediately once
    run_cycle_selective()

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping.")
            break
        except Exception as e:
            logger.error(f"[loop] Error: {e}", exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    main_loop()
