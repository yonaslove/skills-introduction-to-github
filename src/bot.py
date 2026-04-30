import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
import time
import os
import requests
from features import add_ict_features
from ta.volatility import AverageTrueRange

# --- CONFIGURATION ---
SYMBOL_GOLD = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_M1
RISK_PERCENT = 0.02
CONFIDENCE_THRESHOLD = 0.85

import json

# Load secrets
secrets_path = os.path.join(os.path.dirname(__file__), "..", "secrets.json")
with open(secrets_path, "r") as f:
    secrets = json.load(f)

TELEGRAM_TOKEN = secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = secrets["TELEGRAM_CHAT_ID"]

def send_telegram_msg(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
    except:
        pass

def get_data(symbol, n_candles=200):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_candles)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
    return df

def get_safe_lot_size(risk_amount, price_risk):
    if price_risk <= 0: return 0.01
    lot_size = risk_amount / price_risk / 100
    return max(0.01, round(lot_size, 2))

def send_order(symbol, order_type, price, sl, tp, volume, comment):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": order_type,
        "price": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "magic": 123456,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    return mt5.order_send(request)

def manage_open_positions():
    """Manages open trades with a Trailing Stop."""
    positions = mt5.positions_get(symbol=SYMBOL_GOLD)
    if not positions: return
    for pos in positions:
        ticket = pos.ticket
        current_sl = pos.sl
        tick = mt5.symbol_info_tick(SYMBOL_GOLD)
        current_price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
        
        df = get_data(SYMBOL_GOLD, 50)
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range().iloc[-1]
        
        if pos.type == mt5.POSITION_TYPE_BUY:
            new_sl = current_price - (2 * atr)
            if new_sl > current_sl + (0.5 * atr):
                mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": SYMBOL_GOLD, "sl": float(new_sl), "tp": float(pos.tp), "position": ticket})
        elif pos.type == mt5.POSITION_TYPE_SELL:
            new_sl = current_price + (2 * atr)
            if new_sl < current_sl - (0.5 * atr):
                mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "symbol": SYMBOL_GOLD, "sl": float(new_sl), "tp": float(pos.tp), "position": ticket})

# --- STARTUP ---
if not mt5.initialize():
    quit()

script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, "..", "models", "xgb_ict_model.pkl"))
print("✅ AI Master System Online")

try:
    while True:
        gold_df = get_data(SYMBOL_GOLD)
        if gold_df is None: 
            time.sleep(5)
            continue
            
        df_features = add_ict_features(gold_df.copy())
        latest_row = df_features.iloc[[-2]]
        
        # Features & Prediction
        features_list = ['Bullish_FVG', 'Bearish_FVG', 'Bullish_Sweep', 'Bearish_Sweep', 'EQH', 'EQL', 'BOS_Up', 'BOS_Down', 'Bullish_OB', 'Bearish_OB', 'In_Discount', 'In_Premium', 'London_Killzone', 'NY_Killzone', 'Silver_Bullet', 'RSI', 'ATR', 'Trend_Up']
        X = latest_row.reindex(columns=features_list).fillna(0)
        confidence, pred = np.max(model.predict_proba(X)[0]), model.predict(X)[0]
        
        # Macro Check
        news_pause, dxy_trend, pro_sentiment = 0, 0, 0.5
        fund_path = os.path.join(script_dir, "..", "data", "fundamentals.csv")
        if os.path.exists(fund_path):
            fund_df = pd.read_csv(fund_path)
            news_pause, dxy_trend, pro_sentiment = fund_df.iloc[-1]['News_Pause'], fund_df.iloc[-1]['DXY_Trend'], fund_df.iloc[-1]['Pro_Sentiment']
            
        positions = mt5.positions_get(symbol=SYMBOL_GOLD)
        
        if len(positions) > 0:
            manage_open_positions()
        elif news_pause == 0 and confidence >= CONFIDENCE_THRESHOLD:
            is_killzone = latest_row['London_Killzone'].values[0] or latest_row['NY_Killzone'].values[0]
            
            if is_killzone:
                atr = latest_row['ATR'].values[0]
                tick = mt5.symbol_info_tick(SYMBOL_GOLD)
                
                # Context for the message
                sweeps = []
                if latest_row['Bullish_Sweep'].values[0]: sweeps.append("Bullish ERL Sweep")
                if latest_row['Bearish_Sweep'].values[0]: sweeps.append("Bearish ERL Sweep")
                sweep_text = ", ".join(sweeps) if sweeps else "None"

                if pred == 1 and latest_row['Trend_Up'].values[0] and pro_sentiment >= 0.4 and dxy_trend <= 0:
                    price = tick.ask
                    sl, tp = price - (2 * atr), latest_row['ERL_High'].values[0]
                    lots = get_safe_lot_size(mt5.account_info().balance * RISK_PERCENT, price - sl)
                    send_order(SYMBOL_GOLD, mt5.ORDER_TYPE_BUY, price, sl, tp, lots, "MASTER-SNIPER")
                    
                    msg = f"🔔 <b>ELITE BUY SIGNAL</b>\n"
                    msg += f"<b>Symbol:</b> {SYMBOL_GOLD}\n"
                    msg += f"<b>AI Confidence:</b> {confidence:.2%}\n\n"
                    msg += f"📊 <b>Institutional Context:</b>\n"
                    msg += f"• ERL Sweep: {sweep_text}\n"
                    msg += f"• Session: {'London' if latest_row['London_Killzone'].values[0] else 'New York'}\n"
                    msg += f"• HTF Trend: {'UP' if latest_row['Trend_Up'].values[0] else 'DOWN'}\n\n"
                    msg += f"📜 <b>Trade Details:</b>\n"
                    msg += f"• Entry Price: {price:.2f}\n"
                    msg += f"• Stop Loss: {sl:.2f}\n"
                    msg += f"• Take Profit: {tp:.2f}\n"
                    msg += f"• Lot Size: {lots}"
                    send_telegram_msg(msg)
                    
                elif pred == 2 and not latest_row['Trend_Up'].values[0] and pro_sentiment <= 0.6 and dxy_trend >= 0:
                    price = tick.bid
                    sl, tp = price + (2 * atr), latest_row['ERL_Low'].values[0]
                    lots = get_safe_lot_size(mt5.account_info().balance * RISK_PERCENT, sl - price)
                    send_order(SYMBOL_GOLD, mt5.ORDER_TYPE_SELL, price, sl, tp, lots, "MASTER-SNIPER")
                    
                    msg = f"🔔 <b>ELITE SELL SIGNAL</b>\n"
                    msg += f"<b>Symbol:</b> {SYMBOL_GOLD}\n"
                    msg += f"<b>AI Confidence:</b> {confidence:.2%}\n\n"
                    msg += f"📊 <b>Institutional Context:</b>\n"
                    msg += f"• ERL Sweep: {sweep_text}\n"
                    msg += f"• Session: {'London' if latest_row['London_Killzone'].values[0] else 'New York'}\n"
                    msg += f"• HTF Trend: {'UP' if latest_row['Trend_Up'].values[0] else 'DOWN'}\n\n"
                    msg += f"📜 <b>Trade Details:</b>\n"
                    msg += f"• Entry Price: {price:.2f}\n"
                    msg += f"• Stop Loss: {sl:.2f}\n"
                    msg += f"• Take Profit: {tp:.2f}\n"
                    msg += f"• Lot Size: {lots}"
                    send_telegram_msg(msg)

        time.sleep(30)

except KeyboardInterrupt:
    print("🛑 Stopped")
finally:
    mt5.shutdown()
