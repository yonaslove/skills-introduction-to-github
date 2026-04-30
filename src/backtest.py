import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

class SimpleBacktester:
    def __init__(self, initial_capital=10000.0, risk_per_trade=0.01):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        
        # Performance Tracking
        self.trades = []
        self.equity_curve = []
        
        # Hardcoded Risk/Reward settings (Basic)
        # e.g., stop loss = 0.1% away from entry, take profit = 0.2% away
        self.sl_pct = 0.001
        self.tp_pct = 0.002

    def run(self, df, predictions):
        print(f"Starting Backtest with Initial Capital: ${self.initial_capital:,.2f}")
        self.capital = self.initial_capital
        self.trades = []
        self.equity_curve = []
        
        in_trade = False
        trade_type = None
        entry_price = 0.0
        sl_price = 0.0
        tp_price = 0.0
        position_size = 0.0
        
        for i in range(len(df)):
            current_row = df.iloc[i]
            pred = predictions[i]
            
            # Record equity
            self.equity_curve.append(self.capital)
            
            # If we are currently in a trade, check if it hit SL or TP
            if in_trade:
                high = current_row['High']
                low = current_row['Low']
                
                # Check Long Trade
                if trade_type == 'BUY':
                    if low <= sl_price:
                        # Hit Stop Loss
                        loss = position_size * (entry_price - sl_price)
                        self.capital -= loss
                        self._record_trade(current_row.name, 'BUY', entry_price, sl_price, -loss, 'SL')
                        in_trade = False
                    elif high >= tp_price:
                        # Hit Take Profit
                        profit = position_size * (tp_price - entry_price)
                        self.capital += profit
                        self._record_trade(current_row.name, 'BUY', entry_price, tp_price, profit, 'TP')
                        in_trade = False
                        
                # Check Short Trade
                elif trade_type == 'SELL':
                    if high >= sl_price:
                        # Hit Stop Loss
                        loss = position_size * (sl_price - entry_price)
                        self.capital -= loss
                        self._record_trade(current_row.name, 'SELL', entry_price, sl_price, -loss, 'SL')
                        in_trade = False
                    elif low <= tp_price:
                        # Hit Take Profit
                        profit = position_size * (entry_price - tp_price)
                        self.capital += profit
                        self._record_trade(current_row.name, 'SELL', entry_price, tp_price, profit, 'TP')
                        in_trade = False
            
            # If not in trade, look for new signal
            if not in_trade:
                # 1 = BUY, -1 = SELL
                if pred == 1:
                    entry_price = current_row['Close']
                    sl_price = entry_price * (1 - self.sl_pct)
                    tp_price = entry_price * (1 + self.tp_pct)
                    trade_type = 'BUY'
                    in_trade = True
                    
                    # Calculate position size based on risk
                    risk_amount = self.capital * self.risk_per_trade
                    price_risk = entry_price - sl_price
                    position_size = risk_amount / price_risk if price_risk > 0 else 0
                    
                elif pred == -1:
                    entry_price = current_row['Close']
                    sl_price = entry_price * (1 + self.sl_pct)
                    tp_price = entry_price * (1 - self.tp_pct)
                    trade_type = 'SELL'
                    in_trade = True
                    
                    risk_amount = self.capital * self.risk_per_trade
                    price_risk = sl_price - entry_price
                    position_size = risk_amount / price_risk if price_risk > 0 else 0

        # Add remaining capital to equity curve if we end in a trade
        if len(self.equity_curve) < len(df):
            self.equity_curve.append(self.capital)

        self._print_summary()
        return pd.DataFrame(self.trades)

    def _record_trade(self, time, type, entry, exit_price, pnl, result):
        self.trades.append({
            'Time': time,
            'Type': type,
            'Entry': entry,
            'Exit': exit_price,
            'PnL': pnl,
            'Result': result
        })

    def _print_summary(self):
        total_trades = len(self.trades)
        if total_trades == 0:
            print("\nNo trades executed during this period.")
            return
            
        winning_trades = sum(1 for t in self.trades if t['PnL'] > 0)
        win_rate = (winning_trades / total_trades) * 100
        
        gross_profit = sum(t['PnL'] for t in self.trades if t['PnL'] > 0)
        gross_loss = abs(sum(t['PnL'] for t in self.trades if t['PnL'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        net_profit = self.capital - self.initial_capital
        return_pct = (net_profit / self.initial_capital) * 100

        print("\n" + "="*40)
        print("BACKTESTING RESULTS")
        print("="*40)
        print(f"Total Trades:   {total_trades}")
        print(f"Win Rate:       {win_rate:.2f}%")
        print(f"Profit Factor:  {profit_factor:.2f}")
        print(f"Net Profit:     ${net_profit:,.2f} ({return_pct:.2f}%)")
        print(f"Final Capital:  ${self.capital:,.2f}")
        print("="*40)

    def plot_equity_curve(self, save_path=None):
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve, label='Portfolio Equity', color='blue')
        plt.title('Backtest Equity Curve')
        plt.xlabel('Time (Bars)')
        plt.ylabel('Capital ($)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Equity curve saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(script_dir, "..", "features", "xauusd_features.csv")
    model_path = os.path.join(script_dir, "..", "models", "xgb_ict_model.pkl")
    
    if os.path.exists(features_path) and os.path.exists(model_path):
        print("Loading data and model...")
        df = pd.read_csv(features_path, index_col=0, parse_dates=True)
        model = joblib.load(model_path)
        
        # Prepare features exactly as in training
        features = ['Bullish_FVG', 'Bearish_FVG', 'Bullish_Sweep', 'Bearish_Sweep',
                    'EQH', 'EQL', 'BOS_Up', 'BOS_Down', 'Bullish_OB', 'Bearish_OB',
                    'In_Discount', 'In_Premium', 'London_Killzone', 'NY_Killzone', 
                    'Silver_Bullet', 'RSI', 'ATR', 'Trend_Up']
        
        features = [f for f in features if f in df.columns]
            
        # We only backtest on the OOS (Out of Sample) test set
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()
        X_test = test_df[features].fillna(0)
        
        print(f"Generating predictions for {len(test_df)} samples...")
        predictions = model.predict(X_test)
        
        # Run Backtest with Dynamic SL/TP
        class ICTBacktester(SimpleBacktester):
            def run(self, df, predictions):
                print(f"Starting ICT Backtest with Dynamic SL/TP...")
                self.capital = self.initial_capital
                self.trades = []
                self.equity_curve = []
                
                in_trade = False
                trade_type = None
                entry_price = 0.0
                sl_price = 0.0
                tp_price = 0.0
                position_size = 0.0
                
                # Rolling window for dynamic SL
                df['Dynamic_Low'] = df['Low'].rolling(window=10).min()
                df['Dynamic_High'] = df['High'].rolling(window=10).max()
                
                for i in range(len(df)):
                    current_row = df.iloc[i]
                    pred = predictions[i]
                    self.equity_curve.append(self.capital)
                    atr = current_row['ATR'] if 'ATR' in current_row else 0
                    
                    if in_trade:
                        if trade_type == 'BUY':
                            # Trailing Stop: Move SL up if price moves in favor
                            # Lock in profit if price > entry + 1 ATR
                            if current_row['Close'] > entry_price + atr:
                                sl_price = max(sl_price, current_row['Close'] - (2 * atr))
                            
                            if current_row['Low'] <= sl_price:
                                loss = position_size * (entry_price - sl_price)
                                self.capital -= loss
                                self._record_trade(current_row.name, 'BUY', entry_price, sl_price, -loss, 'SL' if sl_price < entry_price else 'TSL')
                                in_trade = False
                            elif current_row['High'] >= tp_price:
                                profit = position_size * (tp_price - entry_price)
                                self.capital += profit
                                self._record_trade(current_row.name, 'BUY', entry_price, tp_price, profit, 'TP')
                                in_trade = False
                        elif trade_type == 'SELL':
                            # Trailing Stop
                            if current_row['Close'] < entry_price - atr:
                                sl_price = min(sl_price, current_row['Close'] + (2 * atr))
                                
                            if current_row['High'] >= sl_price:
                                loss = position_size * (sl_price - entry_price)
                                self.capital -= loss
                                self._record_trade(current_row.name, 'SELL', entry_price, sl_price, -loss, 'SL' if sl_price > entry_price else 'TSL')
                                in_trade = False
                            elif current_row['Low'] <= tp_price:
                                profit = position_size * (entry_price - tp_price)
                                self.capital += profit
                                self._record_trade(current_row.name, 'SELL', entry_price, tp_price, profit, 'TP')
                                in_trade = False
                    
                    if not in_trade:
                        if pred == 1: # BUY
                            entry_price = current_row['Close']
                            # SL below recent low (at least 0.05% away)
                            sl_price = min(current_row['Dynamic_Low'], entry_price * 0.9995)
                            risk = entry_price - sl_price
                            tp_price = entry_price + (risk * 2) # 1:2 RR
                            
                            trade_type = 'BUY'
                            in_trade = True
                            risk_amount = self.capital * self.risk_per_trade
                            position_size = risk_amount / risk if risk > 0 else 0
                            
                        elif pred == 2: # SELL
                            entry_price = current_row['Close']
                            # SL above recent high
                            sl_price = max(current_row['Dynamic_High'], entry_price * 1.0005)
                            risk = sl_price - entry_price
                            tp_price = entry_price - (risk * 2) # 1:2 RR
                            
                            trade_type = 'SELL'
                            in_trade = True
                            risk_amount = self.capital * self.risk_per_trade
                            position_size = risk_amount / risk if risk > 0 else 0

                self._print_summary()
                return pd.DataFrame(self.trades)

        backtester = ICTBacktester(initial_capital=10000, risk_per_trade=0.01)
        trades_df = backtester.run(test_df, predictions)
        
        # Save results
        backtesting_dir = os.path.join(script_dir, "..", "backtesting")
        os.makedirs(backtesting_dir, exist_ok=True)
        if not trades_df.empty:
            trades_df.to_csv(os.path.join(backtesting_dir, "trade_log.csv"), index=False)
            backtester.plot_equity_curve(save_path=os.path.join(backtesting_dir, "equity_curve.png"))
            
    else:
        print("Missing features or model.")
