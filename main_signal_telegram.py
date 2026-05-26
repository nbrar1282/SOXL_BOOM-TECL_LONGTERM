import json
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os
import requests
from datetime import datetime

# Suppress warnings for a clean output
warnings.filterwarnings('ignore')

# ─── TELEGRAM CONFIGURATION ───────────────────────────────────────────────────
# Keep these secure!

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram(text: str):
    """Sends a formatted message to your Telegram chat."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        print("✅ Telegram message sent successfully.")
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")

# ─── STRATEGY ENGINE ──────────────────────────────────────────────────────────
class ComposerLiveEngine:
    def __init__(self, json_path: str, start_date: str = '2017-01-01', end_date: str = None):
        self.start_date = start_date
        self.end_date = end_date
        self.json_path = json_path
        
        # Handle Strategy Loading (Strictly from file)
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as f:
                self.spec = json.load(f)
        else:
            raise FileNotFoundError(f"Could not find strategy file: {self.json_path}")
            
        self.tickers = self._discover_tickers(self.spec)
        self.data = pd.DataFrame()
        self.cache = {}

    def _discover_tickers(self, node) -> set:
        tickers = set()
        if isinstance(node, dict):
            if node.get("step") == "asset" and "ticker" in node:
                tickers.add(node.get("ticker"))
            for key in ["lhs-val", "rhs-val"]:
                val = node.get(key)
                if isinstance(val, str) and val.isalpha() and val.isupper() and 1 <= len(val) <= 5:
                    tickers.add(val)
            for val in node.values():
                tickers.update(self._discover_tickers(val))
        elif isinstance(node, list):
            for item in node:
                tickers.update(self._discover_tickers(item))
        return tickers

    def sync_data(self):
        """Downloads historical buffer + live prices."""
        if not self.tickers: return
        
        # Buffer to warm up the moving averages and max drawdowns natively
        start_buffer = pd.Timestamp(self.start_date) - pd.DateOffset(days=500)
        print(f"\nSyncing market data for {len(self.tickers)} tickers...")
        
        # CRITICAL: auto_adjust=False is required to prevent standard deviation corruption from stock splits
        df = yf.download(list(self.tickers), start=start_buffer, end=self.end_date, progress=False, auto_adjust=False)
        
        if len(self.tickers) == 1:
            ticker = list(self.tickers)[0]
            self.data = pd.DataFrame({ticker: df['Adj Close'] if 'Adj Close' in df.columns else df['Close']})
        else:
            if isinstance(df.columns, pd.MultiIndex):
                self.data = df['Adj Close'] if 'Adj Close' in df.columns.get_level_values(0) else df['Close']
            else:
                self.data = df[['Adj Close']] if 'Adj Close' in df.columns else df[['Close']]

        # CRITICAL: Clean timezones and NaNs to prevent indicator lookup failures
        self.data.index = pd.to_datetime(self.data.index).tz_localize(None)
        self.data = self.data.replace([np.inf, -np.inf], np.nan).ffill().dropna()

    # --- High Fidelity Indicators ---
    def _calc_rsi(self, series, window=14):
        delta = series.diff()
        up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=window-1, adjust=False).mean()
        ema_down = down.ewm(com=window-1, adjust=False).mean()
        rs = ema_up / (ema_down + 1e-10)
        return 100 - (100 / (1 + rs))

    def _calc_cumulative_return(self, series, window):
        return (series / series.shift(window) - 1) * 100

    def _calc_std(self, series, window=20):
        return series.pct_change().rolling(window).std(ddof=1) * 100

    def _calc_max_drawdown(self, series, window=63):
        def get_max_dd(x):
            if len(x) == 0: return 0
            roll_max = np.maximum.accumulate(x)
            return np.max((roll_max - x) / roll_max)
        return series.rolling(window).apply(get_max_dd, raw=True) * 100

    def get_indicator(self, name, ticker, window, date):
        cache_key = (name, ticker, window)
        if cache_key not in self.cache:
            if ticker not in self.data.columns: return np.nan
            series = self.data[ticker]
            if name == "relative-strength-index": self.cache[cache_key] = self._calc_rsi(series, window)
            elif name == "moving-average-price": self.cache[cache_key] = series.rolling(window).mean()
            elif name == "cumulative-return": self.cache[cache_key] = self._calc_cumulative_return(series, window)
            elif name in ["standard-deviation", "standard-deviation-return"]: self.cache[cache_key] = self._calc_std(series, window)
            elif name == "max-drawdown": self.cache[cache_key] = self._calc_max_drawdown(series, window)
            else: self.cache[cache_key] = series
        
        try: return float(self.cache[cache_key].loc[date])
        except: return np.nan

    def _evaluate_condition(self, cond, date):
        try:
            l_win = int(cond.get("lhs-fn-params", {}).get("window", cond.get("lhs-window-days", 14)))
            r_win = int(cond.get("rhs-fn-params", {}).get("window", cond.get("rhs-window-days", 14)))
            
            lhs = self.get_indicator(cond.get("lhs-fn"), cond.get("lhs-val"), l_win, date) if "lhs-fn" in cond else float(cond.get("lhs-val"))
            rhs = float(cond.get("rhs-val")) if cond.get("rhs-fixed-value?") else self.get_indicator(cond.get("rhs-fn"), cond.get("rhs-val"), r_win, date)
            
            if pd.isna(lhs) or pd.isna(rhs): return False
            op = cond.get("comparator")
            
            if op == "gt": return lhs > rhs
            if op == "lt": return lhs < rhs
            if op == "gte": return lhs >= rhs
            if op == "lte": return lhs <= rhs
            return round(lhs, 4) == round(rhs, 4)
        except: return False

    def solve_allocation(self, node, date):
        step = node.get("step")
        if step == "asset": return {node["ticker"]: 1.0}
        
        if step in ["root", "group", "wt-cash-equal", "wt-equal"]:
            children = node.get("children", [])
            if not children: return {}
            combined, weight = {}, 1.0 / len(children)
            for child in children:
                res = self.solve_allocation(child, date)
                for t, w in res.items(): combined[t] = combined.get(t, 0.0) + (w * weight)
            return combined
            
        # Handle explicitly defined weights (Required for TECL UVXY/BIL splits)
        if step == "wt-cash-specified":
            children = node.get("children", [])
            combined = {}
            for child in children:
                weight_node = child.get("weight", {"num": 0, "den": 1})
                weight = float(weight_node["num"]) / float(weight_node["den"])
                res = self.solve_allocation(child, date)
                for t, w in res.items(): combined[t] = combined.get(t, 0.0) + (w * weight)
            return combined

        if step == "if":
            for child in node.get("children", []):
                if not child.get("is-else-condition?") and self._evaluate_condition(child, date):
                    return self.solve_allocation(child, date)
            for child in node.get("children", []):
                if child.get("is-else-condition?"): return self.solve_allocation(child, date)
            return {}
            
        if step == "if-child":
            return self.solve_allocation({"step": "group", "children": node.get("children", [])}, date)
            
        if step == "filter":
            candidates = []
            for child in node.get("children", []): candidates.extend(self.solve_allocation(child, date).keys())
            if not candidates: return {}
            
            sort_fn = node.get("sort-by-fn", "cumulative-return")
            window = int(node.get("sort-by-window-days", 21))
            
            scores = sorted([(t, self.get_indicator(sort_fn, t, window, date)) for t in set(candidates) if not pd.isna(self.get_indicator(sort_fn, t, window, date))], 
                            key=lambda x: x[1], reverse=(node.get("select-fn") == "top"))
            selected = [x[0] for x in scores[:int(node.get("select-n", 1))]]
            return {t: 1.0/len(selected) for t in selected} if selected else {}
            
        return {}


def run_live_report(json_path, start_date):
    engine = ComposerLiveEngine(json_path, start_date)
    engine.sync_data()
    
    history = []
    # CRITICAL: STRICT START DATE ENFORCEMENT
    test_dates = engine.data[engine.data.index >= pd.Timestamp(engine.start_date)].index
    
    print(f"Verifying historical performance strictly from {engine.start_date}...")
    for date in test_dates:
        alloc = engine.solve_allocation(engine.spec, date)
        alloc['Date'] = date
        history.append(alloc)
        
    alloc_df = pd.DataFrame(history).set_index('Date').fillna(0.0)
    
    # Accurate return mapping (shift(-1) applies today's signal to tomorrow's return)
    daily_rets = engine.data.pct_change().shift(-1)
    port_rets = pd.Series(0.0, index=alloc_df.index)
    
    for date in alloc_df.index:
        if date not in daily_rets.index: continue
        weights = alloc_df.loc[date]
        day_ret = 0
        for ticker, weight in weights.items():
            if weight > 0 and ticker in daily_rets.columns:
                r = daily_rets.at[date, ticker]
                if not pd.isna(r): day_ret += r * weight
        port_rets.loc[date] = day_ret

    equity_curve = (1 + port_rets).cumprod()

    # Prepare Telegram Message
    latest_date = engine.data.index[-1]
    current_alloc = engine.solve_allocation(engine.spec, latest_date)
    
    msg = [f"🚀 *Strategy: {engine.spec.get('name', 'Live Signal')}*"]
    msg.append(f"📅 *Signal Date:* {latest_date.date()}")
    msg.append("\n⚖️ *TARGET HOLDINGS FOR TOMORROW:*")
    
    if not current_alloc:
        msg.append("👉 *100% CASH*")
    else:
        for t, w in sorted(current_alloc.items(), key=lambda x: x[1], reverse=True):
            if w > 0.001:
                price = engine.data[t].iloc[-1]
                msg.append(f"🟢 *{t}*: {w:.1%} (`${price:.2f}`)")

    # Yearly Stats Summary
    msg.append("\n📅 *YEARLY PERFORMANCE:*")
    yearly = port_rets.groupby(port_rets.index.year).apply(lambda x: (1 + x).prod() - 1)
    
    grid = []
    years = list(yearly.items())
    for i in range(0, len(years), 2):
        row = [f"`{y}: {r:+.1%}`" for y, r in years[i:i+2]]
        grid.append(" | ".join(row))
    msg.extend(grid)

    msg.append(f"\n📈 *Total Return:* {(equity_curve.iloc[-1] - 1):.1%}")
    msg.append(f"📉 *Max DD:* {((equity_curve / equity_curve.cummax()) - 1).min():.1%}")

    final_text = "\n".join(msg)
    print("\n" + final_text + "\n")
    send_telegram(final_text)

if __name__ == "__main__":
    # Define your files and their respective start dates here
    STRATEGIES = [
        {'file': 's.json', 'start_date': '2013-03-01'},
        {'file': 't.json', 'start_date': '2013-03-01'},
        {'file': 'u.json', 'start_date': '2016-03-01'}
    ]
    
    print("Initializing Multi-Strategy Telegram Update...")
    print("="*50)
    
    for strat in STRATEGIES:
        file_name = strat['file']
        start_date = strat['start_date']
        
        if not os.path.exists(file_name):
            print(f"[!] Skipping {file_name}: File not found in directory.")
            continue
            
        print(f"\n⚙️ PROCESSING FILE: {file_name}")
        run_live_report(file_name, start_date)
        
    print("="*50)
    print("Process Complete.")
