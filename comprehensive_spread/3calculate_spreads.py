import pandas as pd
import numpy as np
import time
from pathlib import Path

def calculate_spread_prices(spreads_file, merged_data_file, output_file):
    """
    Calculate spread prices based on spread list and merged price data
    
    Parameters:
    -----------
    spreads_file : str
        Path to the CSV file containing spread definitions
    merged_data_file : str
        Path to the feather file containing merged price data
    output_file : str
        Path to save the calculated spread prices
    """
    print(f"Loading spread definitions from {spreads_file}")
    spreads_df = pd.read_csv(spreads_file)
    
    print(f"Loading merged price data from {merged_data_file}")
    price_data = pd.read_feather(merged_data_file)
    
    # Extract timestamp column
    timestamps = price_data['timestamp']
    
    # Initialize result dataframe with timestamp column
    result_df = pd.DataFrame({'timestamp': timestamps})
    
    # Track progress
    total_spreads = len(spreads_df)
    print(f"Calculating prices for {total_spreads} spreads...")
    start_time = time.time()
    
    # Process each spread
    for idx, row in spreads_df.iterrows():
        spread_code = row['spread_code']
        contract_a = row['contract_a']
        contract_b = row['contract_b']
        
        # Progress update
        if (idx + 1) % 10 == 0 or (idx + 1) == total_spreads:
            elapsed = time.time() - start_time
            print(f"Processing spread {idx+1}/{total_spreads}: {spread_code} ({elapsed:.2f}s elapsed)")
        
        # Check if both contracts exist in price data
        if contract_a in price_data.columns and contract_b in price_data.columns:
            # Calculate spread price (A - B)
            result_df[spread_code] = price_data[contract_a] - price_data[contract_b]
        else:
            missing = []
            if contract_a not in price_data.columns:
                missing.append(contract_a)
            if contract_b not in price_data.columns:
                missing.append(contract_b)
            print(f"Warning: Cannot calculate {spread_code}, missing contracts: {', '.join(missing)}")
    
    # Save result to feather file
    print(f"Saving {len(result_df.columns)-1} spread prices to {output_file}")
    result_df.to_feather(output_file)
    
    # Print summary
    print("\nCalculation complete!")
    print(f"Total time: {time.time() - start_time:.2f} seconds")
    print(f"Total spreads calculated: {len(result_df.columns)-1}")
    print(f"Time range: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")
    
    return result_df

def generate_spread_stats(spread_prices_file, output_file=None):
    """Generate basic statistics for each spread"""
    print(f"Loading spread prices from {spread_prices_file}")
    df = pd.read_feather(spread_prices_file)
    
    # Calculate statistics
    stats = []
    for col in df.columns:
        if col != 'timestamp':
            stats.append({
                'spread_code': col,
                'count': df[col].count(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing_pct': (df[col].isna().sum() / len(df)) * 100
            })
    
    stats_df = pd.DataFrame(stats)
    
    # Save if output file is specified
    if output_file:
        stats_df.to_csv(output_file, index=False)
        print(f"Spread statistics saved to {output_file}")
    
    return stats_df

if __name__ == "__main__":
    # File paths
    spreads_file = "./iron_spreads.csv"
    merged_data_file = "./merged.feather"
    output_file = "./spread_prices.feather"
    stats_file = "./spread_stats.csv"
    
    # Check if input files exist
    if not Path(spreads_file).exists():
        print(f"Error: Spread list file not found: {spreads_file}")
        exit(1)
    
    if not Path(merged_data_file).exists():
        print(f"Error: Merged data file not found: {merged_data_file}")
        exit(1)
    
    # Calculate spread prices
    spread_prices = calculate_spread_prices(
        spreads_file=spreads_file,
        merged_data_file=merged_data_file,
        output_file=output_file
    )
    
    # Generate statistics
    spread_stats = generate_spread_stats(
        spread_prices_file=output_file,
        output_file=stats_file
    )
    
    # Display sample of results
    print("\nSample of calculated spread prices:")
    sample_cols = ['timestamp'] + list(spread_prices.columns[1:])[:5]  # First 5 spreads
    print(spread_prices[sample_cols].head().to_string())
    
    print("\nSpread statistics summary:")
    print(spread_stats.head(10).to_string()) 