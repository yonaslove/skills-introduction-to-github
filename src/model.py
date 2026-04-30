import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score
import joblib
import os
import matplotlib.pyplot as plt

def create_labels(df, lookahead=10):
    """
    Creates target labels for the model.
    1 (Buy): Price goes up in the next `lookahead` candles.
    -1 (Sell): Price goes down.
    0 (No Trade): Price stays relatively flat.
    """
    print("Creating labels with profit-optimized thresholds...")
    # Calculate future max/min return over lookahead window for better labeling
    future_close = df['Close'].shift(-lookahead)
    df['Future_Return'] = (future_close - df['Close']) / df['Close']
    
    # Define thresholds (0.15% move for 1m timeframe is reasonable)
    threshold = 0.0015
    
    conditions = [
        (df['Future_Return'] >= threshold),
        (df['Future_Return'] <= -threshold)
    ]
    choices = [1, 2] # Use 0, 1, 2 for XGBoost (Label 2 for Sell, 0 for Neutral, 1 for Buy)
    # Actually, let's use 0: Neutral, 1: Buy, 2: Sell
    
    df['Target'] = np.select(conditions, choices, default=0)
    
    # PROFIT TIP: Only train on rows where at least one ICT feature is present
    ict_cols = ['Bullish_FVG', 'Bearish_FVG', 'Bullish_Sweep', 'Bearish_Sweep', 
                'EQH', 'EQL', 'BOS_Up', 'BOS_Down', 'Bullish_OB', 'Bearish_OB']
    
    # Filter for rows with at least one signal
    df_filtered = df[(df[ict_cols].sum(axis=1) > 0)].copy()
    
    print(f"Original samples: {len(df)} | Filtered samples (ICT active): {len(df_filtered)}")
    
    # Drop rows with NaN targets
    df_filtered.dropna(subset=['Future_Return'], inplace=True)
    return df_filtered

def train_model(df):
    """
    Trains an XGBoost classifier using advanced ICT and Technical features.
    """
    print("Preparing data for training...")
    # Select features to use
    features = ['Bullish_FVG', 'Bearish_FVG', 'Bullish_Sweep', 'Bearish_Sweep',
                'EQH', 'EQL', 'BOS_Up', 'BOS_Down', 'Bullish_OB', 'Bearish_OB',
                'In_Discount', 'In_Premium', 'London_Killzone', 'NY_Killzone', 
                'Silver_Bullet', 'RSI', 'ATR', 'Trend_Up']
    
    features = [f for f in features if f in df.columns]
    
    X = df[features]
    y = df['Target']
    
    # Chronological Split
    split_idx = int(len(X) * 0.8)
    X_train_full, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_full, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # PROFIT TIP: Balance the classes in training data
    # Undersample Neutral class to match Buy/Sell counts
    print("Balancing training classes...")
    train_df = X_train_full.copy()
    train_df['Target'] = y_train_full
    
    buy_samples = train_df[train_df['Target'] == 1]
    sell_samples = train_df[train_df['Target'] == 2]
    neutral_samples = train_df[train_df['Target'] == 0]
    
    n_samples = min(len(neutral_samples), (len(buy_samples) + len(sell_samples)) * 2)
    neutral_downsampled = neutral_samples.sample(n=n_samples, random_state=42)
    
    balanced_train = pd.concat([buy_samples, sell_samples, neutral_downsampled]).sample(frac=1, random_state=42)
    X_train = balanced_train[features]
    y_train = balanced_train['Target']
    
    print(f"Balanced training set: {len(X_train)} samples (Buy: {len(buy_samples)}, Sell: {len(sell_samples)}, Neutral: {len(neutral_downsampled)})")

    print(f"Training XGBoost Classifier...")
    # Using XGBoost with basic parameters
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating Model...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision (Buy): {precision_score(y_test, y_pred, labels=[1], average='macro'):.4f}")
    print(f"Precision (Sell): {precision_score(y_test, y_pred, labels=[2], average='macro'):.4f}")
    
    # Feature Importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\nFeature Importance:")
    print(importance)
    
    return model

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(script_dir, "..", "features", "xauusd_features.csv")
    
    if os.path.exists(features_path):
        print(f"Loading {features_path}...")
        df = pd.read_csv(features_path, index_col=0, parse_dates=True)
        
        # Phase 4 & 5: Trade Filtering & Labeling
        df_labeled = create_labels(df, lookahead=15)
        
        # Phase 6: Model Training
        model = train_model(df_labeled)
        
        # Save model
        models_dir = os.path.join(script_dir, "..", "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "xgb_ict_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    else:
        print(f"Could not find {features_path}.")
