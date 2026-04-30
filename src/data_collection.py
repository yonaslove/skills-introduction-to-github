import pandas as pd
from datasets import load_dataset
import os

def download_and_clean_data(dataset_name, save_path):
    print(f"\n--- Processing dataset: {dataset_name} ---")
    try:
        # Load dataset from huggingface
        dataset = load_dataset(dataset_name)
        
        # Convert the 'train' split to pandas DataFrame
        df = dataset['train'].to_pandas()
        print(f"Successfully loaded {len(df)} rows.")
        
        # Basic Cleaning
        print("Cleaning data...")
        df.drop_duplicates(inplace=True)
        
        # Detect time column
        time_col = None
        for col in ['time', 'Date', 'datetime', 'Date/Time', 'timestamp']:
            if col in df.columns:
                time_col = col
                break
                
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)
            df.sort_index(inplace=True)
        else:
            print("Warning: Could not find a time column. Proceeding without datetime index.")
        
        # Forward fill missing values for price data
        df.ffill(inplace=True)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV
        print(f"Saving to {save_path}... (This might take a while for large datasets)")
        df.to_csv(save_path)
        print(f"✅ Data saved successfully to {save_path}")
        
        return df

    except Exception as e:
        print(f"❌ Error downloading or processing dataset: {e}")
        return None

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    
    # Links found in the project's todo.md
    datasets = {
        "ta4500/xauusd": "xauusd_clean.csv",
        "kafka7/xauusd-gold-price-historical-data-2004-2025": "xauusd_2004_2025.csv"
    }
    
    for ds_name, filename in datasets.items():
        save_file = os.path.join(data_dir, filename)
        if not os.path.exists(save_file):
            download_and_clean_data(ds_name, save_file)
        else:
            print(f"\n✅ Dataset {filename} already exists. Skipping download.")
