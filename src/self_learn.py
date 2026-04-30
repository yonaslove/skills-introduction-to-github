import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score

# --- PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
historical_path = os.path.join(script_dir, "..", "features", "xauusd_features.csv")
experience_path = os.path.join(script_dir, "..", "data", "live_experience.csv")
model_path = os.path.join(script_dir, "..", "models", "xgb_ict_model.pkl")

def label_data(df, lookahead=15):
    """Labels the data based on future price action."""
    df['Future_Return'] = (df['Close'].shift(-lookahead) - df['Close']) / df['Close']
    threshold = 0.0015
    conditions = [(df['Future_Return'] >= threshold), (df['Future_Return'] <= -threshold)]
    choices = [1, 2] # 1: Buy, 2: Sell
    df['Target'] = np.select(conditions, choices, default=0)
    df.dropna(subset=['Future_Return'], inplace=True)
    return df

def run_self_learning():
    if not os.path.exists(experience_path):
        print("❌ No live experience data found. Run the bot first!")
        return

    print("🧠 Starting Self-Learning Process...")
    
    # 1. Load Data
    hist_df = pd.read_csv(historical_path, index_col=0, parse_dates=True)
    exp_df = pd.read_csv(experience_path, index_col=0, parse_dates=True)
    
    # 2. Process Experience Data
    print(f"Processing {len(exp_df)} new experience samples...")
    exp_df = label_data(exp_df)
    
    # 3. Combine Historical + Live Knowledge
    combined_df = pd.concat([hist_df, exp_df]).drop_duplicates().sort_index()
    
    # 4. Prepare Features
    features = ['Bullish_FVG', 'Bearish_FVG', 'Bullish_Sweep', 'Bearish_Sweep',
                'EQH', 'EQL', 'BOS_Up', 'BOS_Down', 'Bullish_OB', 'Bearish_OB',
                'In_Discount', 'In_Premium', 'London_Killzone', 'NY_Killzone', 
                'Silver_Bullet', 'RSI', 'ATR', 'Trend_Up']
    
    # Filter for active ICT setups to keep training high-quality
    ict_cols = ['Bullish_FVG', 'Bearish_FVG', 'Bullish_Sweep', 'Bearish_Sweep', 
                'EQH', 'EQL', 'BOS_Up', 'BOS_Down', 'Bullish_OB', 'Bearish_OB']
    df_filtered = combined_df[combined_df[ict_cols].sum(axis=1) > 0].copy()
    
    X = df_filtered[features].fillna(0)
    y = df_filtered['Target']
    
    # 5. Retrain Model
    print(f"Retraining Model on {len(X)} combined samples...")
    new_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05, 
        objective='multi:softprob', num_class=3, eval_metric='mlogloss'
    )
    new_model.fit(X, y)
    
    # 6. Save and Backup
    joblib.dump(new_model, model_path)
    print(f"✅ AI Model successfully updated with live knowledge!")
    
    # Optional: Clear the experience file to avoid redundant retraining
    # os.remove(experience_path)

if __name__ == "__main__":
    run_self_learning()
