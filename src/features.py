import pandas as pd
import numpy as np
import os
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def add_ict_features(df):
    """Full Master ICT Feature Engine - Gold Focused."""
    # 1. Standard Technicals
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['ATR'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['Trend_Up'] = (df['Close'] > df['EMA_50']).astype(int)
    
    # 2. ICT Liquidity & Structure
    df['ERL_High'] = df['High'].rolling(window=100).max().shift(1)
    df['ERL_Low'] = df['Low'].rolling(window=100).min().shift(1)
    
    df['Bullish_FVG'] = (df['Low'] > df['High'].shift(2)).astype(int)
    df['Bearish_FVG'] = (df['High'] < df['Low'].shift(2)).astype(int)
    
    df['Bullish_Sweep'] = (df['Low'] < df['Low'].rolling(window=20).min().shift(1)).astype(int)
    df['Bearish_Sweep'] = (df['High'] > df['High'].rolling(window=20).max().shift(1)).astype(int)
    
    df['EQH'] = (abs(df['High'] - df['High'].shift(1)) < 0.1).astype(int)
    df['EQL'] = (abs(df['Low'] - df['Low'].shift(1)) < 0.1).astype(int)
    
    df['BOS_Up'] = (df['Close'] > df['High'].rolling(window=20).max().shift(1)).astype(int)
    df['BOS_Down'] = (df['Close'] < df['Low'].rolling(window=20).min().shift(1)).astype(int)
    
    # 3. Displacement & Order Blocks
    df['Body_Size'] = abs(df['Close'] - df['Open'])
    df['Avg_Body'] = df['Body_Size'].rolling(window=20).mean()
    df['Displacement'] = (df['Body_Size'] > (df['Avg_Body'] * 2)).astype(int)
    
    df['Bullish_OB'] = ((df['Close'] > df['Open']) & df['Displacement']).astype(int)
    df['Bearish_OB'] = ((df['Close'] < df['Open']) & df['Displacement']).astype(int)
    
    # 4. Premium / Discount
    df['Range_Mid'] = (df['ERL_High'] + df['ERL_Low']) / 2
    df['In_Discount'] = (df['Close'] < df['Range_Mid']).astype(int)
    df['In_Premium'] = (df['Close'] > df['Range_Mid']).astype(int)
    
    # 5. Session Timing
    df['Hour'] = df.index.hour
    df['London_Killzone'] = df['Hour'].between(7, 10).astype(int)
    df['NY_Killzone'] = df['Hour'].between(12, 15).astype(int)
    df['Silver_Bullet'] = (df['Hour'] == 14).astype(int)
    
    return df
