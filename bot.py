import yfinance as yf
import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
import os
import re
import signal
import sys
from openai import OpenAI
from datetime import datetime, timedelta, date  # make sure `date` is imported too
import pytz
from tqdm import tqdm
import time
import random
import robin_stocks.robinhood as r
import schedule
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

RH_USERNAME = "turnerlevey@verizon.net"
RH_PASSWORD = "Antideftaphoric$"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
ET = pytz.timezone("US/Eastern")
UTC = pytz.utc

today = date.today().strftime("%m/%d/%Y")

INITIAL_CAPITAL = 1000
SLIPPAGE = 0.01
MAX_HOLD_DAYS = 10
LOG_CSV = "gpt_trade_log.csv"
TRADES_FILE = "gpt_open_trades.json"


# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
capital = INITIAL_CAPITAL
open_trades = []


# ─────────────────────────────────────────────
# 2. GPT PROMPT FORMATTER
# ─────────────────────────────────────────────

def format_prompt_from_data(data_line, options_data):
    options_str = f"Top Call Options:\n"
    for call in options_data["calls"]:
        options_str += f"- Strike: {call['strike']}, Bid: {call['bid']}, Ask: {call['ask']}, IV: {call['impliedVolatility']}, Vol: {call['volume']}\n"

    options_str += f"\nTop Put Options:\n"
    for put in options_data["puts"]:
        options_str += f"- Strike: {put['strike']}, Bid: {put['bid']}, Ask: {put['ask']}, IV: {put['impliedVolatility']}, Vol: {put['volume']}\n"

    return f"""
        You are a top-tier financial analyst and options strategist with real-time access to U.S. market data.

        Today’s date is **{today}**.

        Your task is to evaluate the stock below and determine whether there is a **high-confidence, directionally biased options trade** to be made today (either a call or a put).
        You will be working to evaluate a number of stocks every day. This is one of many you will be evaluating today.

        If no clear opportunity exists based on the technicals, momentum, sentiment, volume, or options pricing, assign a **low confidence score** and do **not** recommend a trade.

        Only consider strike prices that are near-the-money (within ~5% of the current stock price), unless you clearly justify using a farther OTM strike due to cost, expected volatility, or directional conviction.

        ---

        📈 Stock Data:
        {data_line}

        💼 Options Chain (Expiration: {options_data['expiration']}):
        {options_str}

        ---

        🎯 Output only in this exact format:
        Stock: [Ticker]
        Data Line: [The stock data given above]
        Options Chain: [The options chain data given above]
        Direction: [Call or Put or None]
        Entry Time: [Time or range]
        Options Contract: [Ticker, strike, expiry, C/P] (or 'None' if no trade)
        Expiration Date: [MM/DD/YYYY] (or 'None')
        Entry Price: [$X.XX] (or 'None')
        Take Profit Target: [$] (or 'None') (This must be higher than the entry price)
        Stop Loss Threshold: [$] (or 'None') (This must be lower than the entry price)
        Confidence Score: [0–100]
        Rationale: [1–3 sentences using the stock’s **technical patterns** (SMA, RSI, volume, ATR), **momentum/sentiment**, and **options pricing characteristics** (IV, spread, volume, IV Rank). You must explicitly reference IV Rank and ATR(14) when evaluating option pricing or likelihood of reaching target. Penalize expensive options when IV Rank is high unless there's a strong breakout setup, and penalize trades with low ATR unless justified.]

        ---

        📊 Confidence Score Guidelines:
        - 90–100: Extremely strong setup (clear technical alignment + volume + sentiment + favorable options pricing)
        - 75–89: Strong setup with alignment, but some volatility or minor uncertainty
        - 60–74: Decent opportunity with risks or unclear pricing
        - 40–59: Mixed indicators or unattractive reward-to-risk
        - 0–39: Avoid — unclear trend, low volume, poor liquidity, or overpriced contracts

        ⛔ High bid-ask spreads, deep OTM strikes, low volume, or overpriced premiums should reduce the confidence score significantly, even with strong technical setups.

        🔒 Rules & Constraints:
        - Do not recommend contracts over $10 unless truly exceptional. Justify clearly.
        - Strike price should be within 5% of the current stock price (i.e., near-the-money) unless the contract is extremely cheap and you clearly explain why it's worth selecting.
        - Only use options with at least 10 contracts of volume to ensure tradability.
        - Avoid recommending any contract that is >14 days to expiration from today's date listed above, unless the contract is very cheap and confidence is above 70. In that case, you must explain exactly why the extra time is beneficial.
        - Be granular and realistic — confidence must vary in single-point increments (e.g., 61, 87).
        - Avoid clustering between 70–85. Most trades should fall below 70 unless exceptional.
        - Do not reuse boilerplate language. Make rationale specific to each trade.
        - If no trade is compelling, say “None” and explain why.
        - Favor options with tighter bid-ask spreads, decent volume, and fair implied volatility.
        - Avoid recommending contracts with a very wide spread (>25% of the bid) unless volume is high and price is very low.

        📐 Technical Indicator Guidelines:

        - **RSI**:
        - RSI > 70 = Overbought (possible reversal or continuation if strong trend)
        - RSI < 30 = Oversold (possible bounce or continuation if downtrend)
        - RSI between 30–70 = Neutral

        - **IV Rank**:
        - > 60 = High → Options are expensive. Penalize long premium trades unless strong breakout potential justifies cost.
        - < 30 = Low → Options are cheap. Favors long premium directional trades.
        - 30–60 = Neutral. Consider in context of technicals.

        - **ATR(14)**:
        - High ATR = High volatility. Good for breakout trades or justifying expensive contracts.
        - Low ATR = Limited movement. Penalize trades where premium is large relative to likely move.

        ✳️ Confidence Calibration:
        - Limit 1–2 trades per day above 80.
        - 2–4 trades in the 70–79 range.
        - Most trades should fall between 20–60.

        🧠 Think like a trader allocating real capital with real risk and evaluating 100+ setups.
        """


def format_validation_prompt(top5_trades_text, expiration_lookup):
    expiration_notes = "\n".join(
        [f"- {ticker}: {exp}" for ticker, exp in expiration_lookup.items()]
    )
    
    return f"""
        You are an expert options strategist reviewing the top 5 GPT-generated trade recommendations from earlier today.

        Today’s date is **{today}**.

        The following tickers were analyzed today. You must use the expiration dates exactly as shown here:

        {expiration_notes}

        Your job is to:
        - Identify any invalid trades (based on price logic, spread, volume, IV Rank, DTE rules, etc.)
        - Correct the trade only if a small change would make it valid
        - Remove any trade that cannot be justified
        - Rank and return the **top 3 final trades** (sorted by confidence)

        Strictly enforce the following rules:
        - Take profit must be greater than entry price, which must be greater than stop loss
        - Entry price must be $10 or less
        - Option must have volume ≥ 10
        - Bid-ask spread must be < 25% of the bid
        - No vague rationale — must reference technical indicators and option metrics clearly

        Format output like this:

        🧪 Explanation:

        -- [Explain any changes that were made/re-orders that occured, trades that were removed, or if there were no changes necessary when it comes to the top 5 trades and why in 1-3 sentences.]

        📈 Final Top 3 AI Trades (Validated):

        --- Trade #1: [TICKER] (Confidence: X) ---
        Direction: [Call/Put]
        Contract: [Symbol, strike, expiry, C/P]
        Entry Price: $X.XX
        TP: $X.XX | SL: $X.XX
        Expiration: [MM/DD/YYYY]
        Rationale: [1–3 clear, specific sentences]

        [repeat for Trade #2 and Trade #3]

        ---

        If there are no valid trades then say this on the last line: "No valid trades were identified today."

        Here are the top 5 trades for your review:

        {top5_trades_text}
        """


# ─────────────────────────────────────────────
# 3. GPT CALL
# ─────────────────────────────────────────────

def ask_gpt_for_trade(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────────
# 4. DATA FUNCTIONS
# ─────────────────────────────────────────────

def get_sp500_tickers():
    res = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    return [row.find_all("td")[0].text.strip() for row in table.find_all("tr")[1:]]

def fetch_yahoo_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if len(hist) < 20:
            return None

        price = hist["Close"].iloc[-1]
        sma10 = hist["Close"].rolling(10).mean().iloc[-1]
        sma50 = hist["Close"].rolling(50).mean().iloc[-1]
        sma200 = hist["Close"].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None
        rsi = compute_rsi(hist["Close"], 14)
        volume = hist["Volume"].iloc[-1]
        avg_volume = hist["Volume"].rolling(20).mean().iloc[-1]

        hist['H-L'] = hist['High'] - hist['Low']
        hist['H-PC'] = abs(hist['High'] - hist['Close'].shift(1))
        hist['L-PC'] = abs(hist['Low'] - hist['Close'].shift(1))
        tr = hist[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "sma10": round(sma10, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2) if sma200 else None,
            "rsi14": round(rsi, 2),
            "volume": int(volume),
            "avg_volume": int(avg_volume),
            "atr14": round(atr, 2),
        }
    except:
        return None
    

def fetch_options_chain(ticker):
    stock = yf.Ticker(ticker)
    try:
        expirations = stock.options
        if not expirations:
            return None

        # Filter expirations to only those between 3 and 14 days from today
        today_dt = datetime.today().date()
        valid_exps = [
            exp for exp in expirations
            if 3 <= (datetime.strptime(exp, "%Y-%m-%d").date() - today_dt).days <= 14
        ]

        if not valid_exps:
            return None  # No valid DTE found

        nearest_exp = valid_exps[0]
        opt_chain = stock.option_chain(nearest_exp)
        calls = opt_chain.calls[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']]
        puts = opt_chain.puts[['strike', 'lastPrice', 'bid', 'ask', 'volume', 'impliedVolatility']]
        top_calls = calls.sort_values(by='volume', ascending=False).head(3)
        top_puts = puts.sort_values(by='volume', ascending=False).head(3)

        ivs = list(top_calls['impliedVolatility'].dropna()) + list(top_puts['impliedVolatility'].dropna())
        if len(ivs) < 2:
            iv_rank = None
        else:
            current_iv = sum(ivs) / len(ivs)
            iv_rank = round((current_iv - min(ivs)) / (max(ivs) - min(ivs)) * 100, 1) if max(ivs) > min(ivs) else 0.0

        return {
            "expiration": nearest_exp,
            "calls": top_calls.to_dict(orient='records'),
            "puts": top_puts.to_dict(orient='records'),
            "iv_rank": iv_rank
        }
    except Exception as e:
        print(f"❌ Error fetching chain for {ticker}: {e}")
        return None
    

def compute_rsi(series, period):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs)).iloc[-1]

# ─────────────────────────────────────────────
# 5. CONFIDENCE PARSER
# ─────────────────────────────────────────────

def extract_confidence_score(text):
    match = re.search(r"Confidence Score:\s*(\d+)", text)
    return int(match.group(1)) if match else 0

# ─────────────────────────────────────────────
# 6. MAIN LOOP
# ─────────────────────────────────────────────

def simulate_entry(contract, take_profit, stop_loss, confidence):
    global capital

    try:
        # Fetch option instruments from Robinhood
        options = r.options.find_options_by_expiration_and_strike(
            contract["ticker"],
            contract["expiry"],
            str(contract["strike"]),
            optionType="call" if contract["type"] == "C" else "put"
        )

        if not options:
            print(f"❌ No Robinhood options found for {contract}")
            return None

        option_id = options[0]["id"]

        # Get market data
        market_data = r.options.get_option_market_data_by_id(option_id)
        if not market_data:
            print(f"❌ No market data for {option_id}")
            return None

        market = market_data[0]
        bid = float(market.get("bid_price") or 0)
        ask = float(market.get("ask_price") or 0)

        # Entry price fallback logic: midpoint > bid/ask avg > last
        if bid > 0 and ask > 0:
            entry_price = round((bid + ask) / 2, 3)
        else:
            print(f"❌ No valid price available for entry on {contract}")
            return None

        cost = entry_price * 100 * (1 + SLIPPAGE)
        if cost > capital:
            print(f"❌ Not enough capital to enter trade on {contract['ticker']}")
            return None

        trade = {
            "ticker": contract["ticker"],
            "type": contract["type"],  # 'C' or 'P'
            "strike": contract["strike"],
            "expiry": contract["expiry"],
            "entry_price": entry_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "entry_time": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
            "confidence": confidence,
            "status": "OPEN"
        }

        capital -= cost
        open_trades.append(trade)
        save_state()

        body_text = f"\n🟢 Entered {trade['ticker']} [{trade['type']}] at ${entry_price} (TP: ${take_profit}, SL: ${stop_loss})"
        print(body_text)

        send_alert_via_discord(
            subject="Exit Alert",
            body=body_text
        )
        
        return trade

    except Exception as e:
        print(f"❌ simulate_entry failed for {contract['ticker']}: {e}")
        return None

def simulate_exit(trade):
    global capital
    ticker = trade["ticker"]
    try:
        options = r.options.find_options_by_expiration_and_strike(
            ticker, trade["expiry"], str(trade["strike"]), optionType="call" if trade["type"] == "C" else "put"
        )
        if not options:
            return False

        opt = options[0]
        market = r.options.get_option_market_data_by_id(opt["id"])[0]
        bid = float(market.get("bid_price") or 0)
        ask = float(market.get("ask_price") or 0)
        mid = (bid + ask) / 2

        exit_price = mid * (1 - SLIPPAGE) if trade["type"] == "C" else mid * (1 + SLIPPAGE)
        entry = trade["entry_price"]
        pnl = (exit_price - entry) * 100
        roi = (pnl / (entry * 100)) * 100

        if (
            exit_price >= trade["take_profit"]
            or exit_price <= trade["stop_loss"]
            or datetime.strptime(trade["expiry"], "%Y-%m-%d").date() <= datetime.now(ET).date()
        ):
            trade.update({
                "exit_price": round(exit_price, 3),
                "exit_time": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
                "pnl": round(pnl, 2),
                "roi": round(roi, 2),
                "status": "CLOSED"
            })
            capital += exit_price * 100
            open_trades.remove(trade)
            log_trade(trade)
            save_state()

            body_text = f"🔴 Exited {trade['ticker']} [{trade['type']}] | PnL: ${pnl:.2f} | ROI: {roi:.2f}%"
            
            print(body_text)
            send_alert_via_discord(
                subject="Entry Alert",
                body=body_text
            )
            return True
        else:
            print(f"TP/SL not yet reached for {trade['ticker']} [{trade['type']}] (${exit_price}, PnL: ${pnl:.2f}, ROI: {roi:.2f}%). Checking again in 5 minutes...")
            return True

    except Exception as e:
        print(f"❌ Failed exit for {ticker}: {e}")
    return False

def log_trade(trade):
    import csv
    file_exists = os.path.exists(LOG_CSV)
    with open(LOG_CSV, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=trade.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(trade)

def save_state():
    with open(TRADES_FILE, "w") as f:
        json.dump({
            "capital": capital,
            "open_trades": open_trades
        }, f, indent=2)

def load_state():
    global capital, open_trades
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "r") as f:
            data = json.load(f)
            capital = data["capital"]
            open_trades = data["open_trades"]

def run_daily_entry():
    tickers = random.sample(get_sp500_tickers(), 70)
    results = []

    print(f"{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} 🔍 Evaluating {len(tickers)} tickers...\n")

    for t in tqdm(tickers, desc="📊 Analyzing"):
        d = fetch_yahoo_data(t)
        options = fetch_options_chain(t)

        if not d or not options or isinstance(options, str):
            tqdm.write(f"❌ Skipping {t} (data or options missing)")
            continue

        data_line = (
            f"{d['ticker']} | Price: ${d['price']} | SMA10: {d['sma10']} | SMA50: {d['sma50']} | "
            f"SMA200: {d['sma200']} | RSI14: {d['rsi14']} | Vol: {d['volume']} | AvgVol: {d['avg_volume']} | "
            f"IV Rank: {options['iv_rank']} | ATR(14): {d['atr14']}"
        )

        prompt = format_prompt_from_data(data_line, options)
        result = ask_gpt_for_trade(prompt)

        confidence = extract_confidence_score(result)
        results.append({"ticker": d["ticker"], "text": result, "confidence": confidence})

    sorted_trades = sorted(results, key=lambda x: x["confidence"], reverse=True)
    print("\n📋 All AI Trade Ideas (Sorted by Confidence):\n")
    for i, trade in enumerate(sorted_trades, 1):
        print(f"--- Trade #{i}: {trade['ticker']} (Confidence: {trade['confidence']}) ---\n{trade['text']}\n")

    expiration_lookup = {t["ticker"]: options["expiration"] for t in sorted_trades[:5]}

    # Take top 5 trades and combine their text
    top5_text = "\n\n".join([t["text"] for t in sorted_trades[:5]])
    validation_prompt = format_validation_prompt(top5_text, expiration_lookup)

    # Ask GPT to validate and refine top 5 into top 3
    final_output = ask_gpt_for_trade(validation_prompt)

    body_text = f"🧠 Post-Validation Final Top 3 Trades:\n\n{final_output}"
            
    print(body_text)
    send_alert_via_discord(
        subject="Trade Options For Today",
        body=body_text
    )

    # Parse and simulate only the top trade
    top_trade_block = final_output.split("--- Trade #1:")[1] if "--- Trade #1:" in final_output else None

    if top_trade_block:
        try:
            lines = top_trade_block.strip().splitlines()
            ticker = lines[0].split()[0].strip()
            confidence = int(re.search(r"Confidence:\s*(\d+)", top_trade_block).group(1))
            direction = re.search(r"Direction:\s*(Call|Put)", top_trade_block).group(1)
            cp = 'C' if direction == "Call" else 'P'

            contract_match = re.search(r"Contract:\s*([\w\d\-\.]+),\s*([\d\.]+),\s*([\d\-]+),\s*(C|P)", top_trade_block)
            if contract_match:
                _, strike, expiry, _ = contract_match.groups()
                strike = float(strike)

                if expiry != expiration_lookup.get(ticker):
                    print(f"⚠️ GPT changed expiration from {expiration_lookup[ticker]} to {expiry} — using original.")
                    expiry = expiration_lookup.get(ticker)

                tp = float(re.search(r"TP:\s*\$(\d+\.\d+)", top_trade_block).group(1))
                sl = float(re.search(r"SL:\s*\$(\d+\.\d+)", top_trade_block).group(1))

                contract = {
                    "ticker": ticker,
                    "strike": strike,
                    "expiry": expiry,
                    "type": cp
                }

                simulate_entry(contract, tp, sl, confidence)
            else:
                print("❌ Could not extract contract line from top trade block.")
        except Exception as e:
            print(f"❌ Error parsing/simulating top trade: {e}")
    else:
        print("❌ No valid top trade block found in GPT output.")

def run_exit_checks():
    if is_market_open_now():
        print(f"{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ⏱️  Checking exits...")
        for trade in open_trades[:]:
            simulate_exit(trade)

        if len(open_trades) == 0:
            print(f"\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} 💤 No open trades to close.\n")

def send_alert_via_discord(
    subject: str,
    body: str,
    webhook_url: str = "https://discord.com/api/webhooks/1380608363754295316/rHD8hZVAetkCM5Is8w8-BwXtrmuh23Shi8GAY8DJJ0Q4SFVgBRAjlkmEmUvhUyySMrue"  # Replace with your webhook URL
):
    """
    Send an alert message to a Discord channel via webhook.
    - recipients: Optional, used just for logging who the alert is for.
    - subject: Title or prefix for the alert.
    - body: Message body.
    - webhook_url: Discord webhook URL.
    """

    content = f"📢 **{subject}**\n\n{body}\n\u200B\n"
    payload = {
        "username": "Alert Bot",
        "content": content
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} 📩 Alert sent to Discord")
    except requests.exceptions.RequestException as e:
        print(f"\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ❌ Failed to send Discord alert: {e}")

def is_market_open_now():
    now_et = datetime.now(ET)
    is_weekday = now_et.weekday() < 5  # 0 = Monday, 4 = Friday
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return is_weekday and market_open <= now_et <= market_close
   
# ─────────────────────────────────────────────
# 7. STARTUP
# ─────────────────────────────────────────────

def shutdown(signum, frame):
    print(f"\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} 👋 Shutting down gracefully…")
    sys.exit(0)

def main():
    global capital
    # 1) Log in once at startup
    print(f"\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ⚙️  Current Version: 1.0.2\n")
    print(f"\n{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} 🔐 Logging into Robinhood…")
    r.authentication.login(RH_USERNAME, RH_PASSWORD)
    print(f"{datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ✅ Logged in.")

    load_state()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Schedule entry at 09:00 ET (08:00 CST)
    schedule.every().day.at("15:05").do(run_daily_entry)

    # Schedule exit checks every 15 minutes
    schedule.every(5).minutes.do(run_exit_checks)

    print(f"\n🚀 GPT Directional Bot Running — Capital: ${capital:.2f}\n")

    while True:
        schedule.run_pending()
        time.sleep(5)

if __name__ == "__main__":
    main()
