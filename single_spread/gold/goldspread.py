import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import sys
import os

def resample_to_15min(df):
    """Convert 1-minute AU data to 15-minute intervals"""
    # Ensure datetime column is datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    # Resample to 15 minutes
    resampled = df.resample('15min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'total_turnover': 'sum',
        'open_interest': 'last'
    })
    # Reset index
    resampled.reset_index(inplace=True)
    return resampled

def adjust_time_zone(df, hours_diff):
    """Adjust timezone, hours_diff is the difference between target and current timezone"""
    df['datetime'] = df['datetime'] - timedelta(hours=hours_diff)
    return df

def filter_by_date_range(df, start_date=None, end_date=None):
    """Filter dataframe by date range"""
    if start_date:
        df = df[df['datetime'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['datetime'] <= pd.to_datetime(end_date)]
    
    if len(df) == 0:
        raise ValueError("No data points in the specified date range")
        
    return df

def load_and_process_au_data(filename, start_date=None, end_date=None):
    """Load and process AU data"""
    print(f"Loading AU data: {filename}")
    try:
        # Read AU data
        au_data = pd.read_csv(filename)
        # Resample to 15-minute intervals
        au_data = resample_to_15min(au_data)
        # Adjust timezone (UTC+8 to UTC+3, 5 hours difference)
        au_data = adjust_time_zone(au_data, 5)
        
        # Filter by date range if specified
        if start_date or end_date:
            au_data = filter_by_date_range(au_data, start_date, end_date)
            
        return au_data
    except Exception as e:
        print(f"Error processing AU data: {e}")
        sys.exit(1)

def load_and_process_xauusd_data(filename, start_date=None, end_date=None):
    """Load and process XAUUSD data"""
    print("Loading XAUUSD data: XAUUSD_M15.csv")
    try:
        # Read XAUUSD data, note tab delimiter
        xauusd_data = pd.read_csv(filename, sep='\t')
        # Merge date and time columns to datetime
        xauusd_data['datetime'] = pd.to_datetime(xauusd_data['<DATE>'] + ' ' + xauusd_data['<TIME>'], 
                                               format='%Y.%m.%d %H:%M:%S')
        # Rename columns
        xauusd_data = xauusd_data.rename(columns={
            '<OPEN>': 'open',
            '<HIGH>': 'high',
            '<LOW>': 'low',
            '<CLOSE>': 'close',
            '<TICKVOL>': 'tickvol',
            '<VOL>': 'volume',
            '<SPREAD>': 'spread'
        })
        # Select required columns
        xauusd_data = xauusd_data[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        # Filter by date range if specified
        if start_date or end_date:
            xauusd_data = filter_by_date_range(xauusd_data, start_date, end_date)
            
        return xauusd_data
    except Exception as e:
        print(f"Error processing XAUUSD data: {e}")
        sys.exit(1)

def load_and_process_usdcnh_data(filename, start_date=None, end_date=None):
    """Load and process USDCNH data for exchange rate"""
    print("Loading USDCNH data: USDCNH_M15.csv")
    try:
        # Read USDCNH data, note tab delimiter
        usdcnh_data = pd.read_csv(filename, sep='\t')
        # Merge date and time columns to datetime
        usdcnh_data['datetime'] = pd.to_datetime(usdcnh_data['<DATE>'] + ' ' + usdcnh_data['<TIME>'], 
                                              format='%Y.%m.%d %H:%M:%S')
        # Rename columns
        usdcnh_data = usdcnh_data.rename(columns={
            '<OPEN>': 'open',
            '<HIGH>': 'high',
            '<LOW>': 'low',
            '<CLOSE>': 'close',
            '<TICKVOL>': 'tickvol',
            '<VOL>': 'volume',
            '<SPREAD>': 'spread'
        })
        # Select required columns
        usdcnh_data = usdcnh_data[['datetime', 'open', 'high', 'low', 'close']]
        
        # Filter by date range if specified
        if start_date or end_date:
            usdcnh_data = filter_by_date_range(usdcnh_data, start_date, end_date)
            
        return usdcnh_data
    except Exception as e:
        print(f"Error processing USDCNH data: {e}")
        sys.exit(1)

def calculate_spread(au_data, xauusd_data, usdcnh_data):
    """Calculate spread using real-time exchange rates
    AU price unit is CNY/gram, XAUUSD price unit is USD/oz
    1 oz = 31.1035 grams
    """
    print("Calculating spread with real-time exchange rates...")
    try:
        # First merge AU and USDCNH data to get real-time exchange rates
        merged_tmp = pd.merge_asof(
            au_data.sort_values('datetime'), 
            usdcnh_data.sort_values('datetime'), 
            on='datetime', 
            direction='nearest',
            suffixes=('_au', '_usdcnh')
        )
        
        # Then merge with XAUUSD data
        merged = pd.merge_asof(
            merged_tmp.sort_values('datetime'), 
            xauusd_data.sort_values('datetime'), 
            on='datetime',
            direction='nearest'
        )
        
        if len(merged) == 0:
            raise ValueError("No matching data points in the specified date range")
        
        # Using real-time USDCNH exchange rates
        print("Using real-time USDCNH exchange rates for calculations")
        
        # AU is already in CNY/gram, keep as is
        # Convert XAUUSD from USD/oz to USD/gram
        merged['xau_usd_per_gram'] = merged['close'] / 31.1035
        
        # Convert XAUUSD from USD/gram to CNY/gram
        merged['xau_cny_per_gram'] = merged['xau_usd_per_gram'] * merged['close_usdcnh']
        
        # Convert AU from CNY/gram to USD/gram
        merged['au_usd_per_gram'] = merged['close_au'] / merged['close_usdcnh']
        
        # Calculate spreads (per gram)
        # USD-denominated spread (AU - XAUUSD) in USD/gram
        merged['spread_usd_per_gram'] = merged['au_usd_per_gram'] - merged['xau_usd_per_gram']
        
        # CNY-denominated spread (AU - XAUUSD) in CNY/gram
        merged['spread_cny_per_gram'] = merged['close_au'] - merged['xau_cny_per_gram']
        
        # Calculate spread percentages
        # USD-denominated spread percentage
        merged['spread_pct_usd'] = merged['spread_usd_per_gram'] / merged['xau_usd_per_gram'] * 100
        
        # CNY-denominated spread percentage
        merged['spread_pct_cny'] = merged['spread_cny_per_gram'] / merged['xau_cny_per_gram'] * 100
        
        # Calculate USD-denominated spread statistics
        spread_usd_mean = merged['spread_usd_per_gram'].mean()
        spread_usd_std = merged['spread_usd_per_gram'].std()
        spread_usd_min = merged['spread_usd_per_gram'].min()
        spread_usd_max = merged['spread_usd_per_gram'].max()
        
        # Calculate CNY-denominated spread statistics
        spread_cny_mean = merged['spread_cny_per_gram'].mean()
        spread_cny_std = merged['spread_cny_per_gram'].std()
        spread_cny_min = merged['spread_cny_per_gram'].min()
        spread_cny_max = merged['spread_cny_per_gram'].max()
        
        # Calculate percentage spread statistics
        spread_pct_usd_mean = merged['spread_pct_usd'].mean()
        spread_pct_cny_mean = merged['spread_pct_cny'].mean()
        
        # Print statistics
        print("\nUSD-denominated Spread Statistics:")
        print(f"  Mean: {spread_usd_mean:.4f} USD/gram ({spread_pct_usd_mean:.2f}%)")
        print(f"  Std Dev: {spread_usd_std:.4f} USD/gram")
        print(f"  Min: {spread_usd_min:.4f} USD/gram")
        print(f"  Max: {spread_usd_max:.4f} USD/gram")
        
        print("\nCNY-denominated Spread Statistics:")
        print(f"  Mean: {spread_cny_mean:.4f} CNY/gram ({spread_pct_cny_mean:.2f}%)")
        print(f"  Std Dev: {spread_cny_std:.4f} CNY/gram")
        print(f"  Min: {spread_cny_min:.4f} CNY/gram")
        print(f"  Max: {spread_cny_max:.4f} CNY/gram")
        
        print(f"\nData points: {len(merged)}")
        print(f"Date range: {merged['datetime'].min()} to {merged['datetime'].max()}")
        print(f"Average USDCNH rate: {merged['close_usdcnh'].mean():.4f}")
        
        return merged
    except Exception as e:
        print(f"Error calculating spread: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def plot_spread(merged_data, contract_name, start_date=None, end_date=None):
    """Plot spread chart with both USD and CNY denominated spreads"""
    print("Creating spread charts...")
    try:
        # Create a 2x2 grid of plots
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(16, 12))
        
        # Sort data by time
        merged_data_sorted = merged_data.sort_values('datetime').copy()
        
        # 1. Combined USD and CNY Spread with dual y-axes (top-left)
        ax1 = plt.subplot(221)
        
        # Plot USD spread on left y-axis
        color1 = 'r'
        ax1.scatter(merged_data_sorted['datetime'], merged_data_sorted['spread_usd_per_gram'], 
                  color=color1, s=3, alpha=0.7, label='USD Spread')
        ax1.set_ylabel('Spread (USD/gram)', color=color1, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Add zero line for USD
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add mean line for USD
        mean_spread_usd = merged_data['spread_usd_per_gram'].mean()
        ax1.axhline(y=mean_spread_usd, color=color1, linestyle='--', alpha=0.5,
                   label=f'USD Mean: {mean_spread_usd:.4f}')
        
        # Create a second y-axis for CNY spread
        ax1_twin = ax1.twinx()
        color2 = 'g'
        ax1_twin.scatter(merged_data_sorted['datetime'], merged_data_sorted['spread_cny_per_gram'], 
                       color=color2, s=3, alpha=0.7, label='CNY Spread')
        ax1_twin.set_ylabel('Spread (CNY/gram)', color=color2, fontsize=12)
        ax1_twin.tick_params(axis='y', labelcolor=color2)
        
        # Add mean line for CNY
        mean_spread_cny = merged_data['spread_cny_per_gram'].mean()
        ax1_twin.axhline(y=mean_spread_cny, color=color2, linestyle='--', alpha=0.5,
                       label=f'CNY Mean: {mean_spread_cny:.4f}')
        
        # Create title
        title = f'AU Contract {contract_name} vs XAUUSD Spread (Dual Currency)'
        if start_date or end_date:
            date_range = ""
            if start_date:
                date_range += f"From: {start_date} "
            if end_date:
                date_range += f"To: {end_date}"
            title += f"\n{date_range}"
        
        ax1.set_title(title, fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Format dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        # 2. USDCNH Exchange Rate (top-right)
        ax2 = plt.subplot(222)
        ax2.scatter(merged_data_sorted['datetime'], merged_data_sorted['close_usdcnh'], 
                  color='purple', s=3, alpha=0.7, label='USDCNH Rate')
        
        # Calculate USDCNH rate statistics
        usdcnh_mean = merged_data['close_usdcnh'].mean()
        usdcnh_std = merged_data['close_usdcnh'].std()
        usdcnh_min = merged_data['close_usdcnh'].min()
        usdcnh_max = merged_data['close_usdcnh'].max()
        
        # Add mean line
        ax2.axhline(y=usdcnh_mean, color='purple', linestyle='--', alpha=0.5, 
                   label=f'Mean: {usdcnh_mean:.4f}')
        
        ax2.set_title(f'USDCNH Exchange Rate (Mean: {usdcnh_mean:.4f})', fontsize=14)
        ax2.set_ylabel('USDCNH Rate', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        
        # Format dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        
        # 3. USD-denominated Price Comparison (bottom-left)
        ax3 = plt.subplot(223)
        ax3.scatter(merged_data_sorted['datetime'], merged_data_sorted['au_usd_per_gram'], 
                  color='g', s=3, alpha=0.7, label='AU (USD/gram)')
        ax3.scatter(merged_data_sorted['datetime'], merged_data_sorted['xau_usd_per_gram'], 
                  color='b', s=3, alpha=0.7, label='XAUUSD (USD/gram)')
        
        ax3.set_ylabel('Price (USD/gram)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        
        # Calculate correlation
        corr_usd = merged_data['au_usd_per_gram'].corr(merged_data['xau_usd_per_gram'])
        ax3.set_title(f'Price Comparison (USD) - Correlation: {corr_usd:.4f}', fontsize=14)
        
        # Format dates
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        
        # 4. CNY-denominated Price Comparison (bottom-right)
        ax4 = plt.subplot(224)
        ax4.scatter(merged_data_sorted['datetime'], merged_data_sorted['close_au'], 
                  color='g', s=3, alpha=0.7, label='AU (CNY/gram)')
        ax4.scatter(merged_data_sorted['datetime'], merged_data_sorted['xau_cny_per_gram'], 
                  color='b', s=3, alpha=0.7, label='XAUUSD (CNY/gram)')
        
        ax4.set_ylabel('Price (CNY/gram)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc='best')
        
        # Calculate correlation
        corr_cny = merged_data['close_au'].corr(merged_data['xau_cny_per_gram'])
        ax4.set_title(f'Price Comparison (CNY) - Correlation: {corr_cny:.4f}', fontsize=14)
        
        # Format dates
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax4.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        
        # Adjust layout
        fig.autofmt_xdate()
        plt.tight_layout()
        
        # Create output directory
        os.makedirs('spread_charts', exist_ok=True)
        
        # Save chart
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_parts = [contract_name, "spread", "pergram", "dual_axis"]
        
        if start_date:
            # Clean date format for filename
            clean_start = start_date.replace("-", "").replace(" ", "").replace(":", "")
            filename_parts.append(f"from{clean_start}")
            
        if end_date:
            # Clean date format for filename
            clean_end = end_date.replace("-", "").replace(" ", "").replace(":", "")
            filename_parts.append(f"to{clean_end}")
            
        filename_parts.append(timestamp)
        
        output_file = f'spread_charts/{"_".join(filename_parts)}.png'
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Chart saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error creating chart: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def validate_date_format(date_str):
    """Validate date string format"""
    if not date_str.strip():
        return None
        
    try:
        # Try to parse the date
        date = pd.to_datetime(date_str)
        # Return standardized format
        return date.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        print(f"Invalid date format: '{date_str}'. Please use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS format.")
        return None

def main():
    print("=" * 60)
    print("AU vs XAUUSD Spread Analysis Tool (with Real-time Exchange Rates)")
    print("=" * 60)
    
    # Get user input for contract
    if len(sys.argv) > 1:
        contract_name = sys.argv[1]
    else:
        contract_name = input("Enter AU contract code (e.g., AU2312): ")
    
    # Get date range from user
    start_date = None
    end_date = None
    
    date_range_needed = input("Do you want to analyze a specific date range? (y/n, default: n): ").lower().strip()
    if date_range_needed == 'y' or date_range_needed == 'yes':
        while True:
            start_date_input = input("Enter start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS, leave blank for earliest): ")
            start_date = validate_date_format(start_date_input) if start_date_input.strip() else None
            
            end_date_input = input("Enter end date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS, leave blank for latest): ")
            end_date = validate_date_format(end_date_input) if end_date_input.strip() else None
            
            if (start_date or start_date_input == '') and (end_date or end_date_input == ''):
                break
    
    # Build AU data filename
    au_filename = f"{contract_name}.csv"
    
    # Check if files exist
    if not os.path.exists(au_filename):
        print(f"Error: AU contract data file {au_filename} not found")
        return
        
    if not os.path.exists("XAUUSD_M15.csv"):
        print("Error: XAUUSD data file XAUUSD_M15.csv not found")
        return
    
    if not os.path.exists("USDCNH_M15.csv"):
        print("Error: USDCNH data file USDCNH_M15.csv not found")
        return
    
    # Load and process data
    au_data = load_and_process_au_data(au_filename, start_date, end_date)
    xauusd_data = load_and_process_xauusd_data("XAUUSD_M15.csv", start_date, end_date)
    usdcnh_data = load_and_process_usdcnh_data("USDCNH_M15.csv", start_date, end_date)
    
    # Calculate spread using real-time exchange rates
    merged_data = calculate_spread(au_data, xauusd_data, usdcnh_data)
    
    # Plot spread chart
    output_file = plot_spread(merged_data, contract_name, start_date, end_date)
    
    print(f"Analysis complete! Spread chart saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
