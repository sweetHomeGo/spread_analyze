import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# 读取数据文件
def load_data(file_path):
    # 读取CSV文件，使用制表符分隔符
    df = pd.read_csv(file_path, delimiter='\t')
    
    # 合并日期和时间列
    df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
    
    # 提取需要的列并重命名
    result_df = df[['datetime', '<CLOSE>']].copy()
    result_df.rename(columns={'<CLOSE>': 'close'}, inplace=True)
    
    return result_df

# 主函数
def analyze_oil_spread():
    # 加载数据
    print("Loading WTI oil data...")
    wti_df = load_data('XTIUSD.csv')
    
    print("Loading Brent oil data...")
    brent_df = load_data('XBRUSD.csv')
    
    # 检查两个数据集的时间范围
    print(f"WTI data time range: {wti_df['datetime'].min()} to {wti_df['datetime'].max()}")
    print(f"Brent data time range: {brent_df['datetime'].min()} to {brent_df['datetime'].max()}")
    
    # 检查采样频率
    wti_intervals = wti_df['datetime'].diff().value_counts()
    brent_intervals = brent_df['datetime'].diff().value_counts()
    print("WTI sampling interval statistics:")
    print(wti_intervals.head())
    print("Brent sampling interval statistics:")
    print(brent_intervals.head())
    
    # 设置时间索引
    wti_df.set_index('datetime', inplace=True)
    brent_df.set_index('datetime', inplace=True)
    
    # 找出共同的时间范围
    start_time = max(wti_df.index.min(), brent_df.index.min())
    end_time = min(wti_df.index.max(), brent_df.index.max())
    print(f"Common time range: {start_time} to {end_time}")
    
    # 过滤数据到共同时间范围
    wti_filtered = wti_df.loc[start_time:end_time]
    brent_filtered = brent_df.loc[start_time:end_time]
    
    # 找出两个数据集共有的时间点
    common_times = wti_filtered.index.intersection(brent_filtered.index)
    print(f"Number of common time points: {len(common_times)}")
    
    # 使用共同的时间点
    wti_aligned = wti_filtered.loc[common_times]
    brent_aligned = brent_filtered.loc[common_times]
    
    # 合并数据
    merged_df = pd.DataFrame({
        'close_wti': wti_aligned['close'],
        'close_brent': brent_aligned['close']
    })
    
    # 计算价差
    merged_df['spread'] = merged_df['close_brent'] - merged_df['close_wti']
    
    # 计算滚动相关系数 (60天窗口，假设每天96个15分钟数据点)
    window_size = 96 * 60  # 60天，每天96个15分钟数据点
    
    # 确保窗口大小不超过数据长度的一半
    if window_size > len(merged_df) // 2:
        window_size = len(merged_df) // 4
        print(f"Adjusted rolling window size to {window_size} data points due to data length constraints")
    
    # 计算滚动相关系数
    merged_df['rolling_corr'] = merged_df['close_wti'].rolling(window=window_size).corr(merged_df['close_brent'])
    
    # 重置索引，使datetime成为列
    merged_df.reset_index(inplace=True)
    
    # 基本统计分析
    print("\nSpread Statistics:")
    spread_stats = {
        "Mean Spread": merged_df['spread'].mean(),
        "Max Spread": merged_df['spread'].max(),
        "Min Spread": merged_df['spread'].min(),
        "Standard Deviation": merged_df['spread'].std(),
        "Median": merged_df['spread'].median()
    }
    
    for stat, value in spread_stats.items():
        print(f"{stat}: {value:.2f}")
    
    # 相关系数统计
    print("\nRolling Correlation Statistics:")
    corr_stats = {
        "Mean Correlation": merged_df['rolling_corr'].mean(),
        "Max Correlation": merged_df['rolling_corr'].max(),
        "Min Correlation": merged_df['rolling_corr'].min(),
        "Standard Deviation": merged_df['rolling_corr'].std()
    }
    
    for stat, value in corr_stats.items():
        print(f"{stat}: {value:.4f}")
    
    # 数据可视化
    print("\nCreating visualization charts...")
    
    # 设置图表风格
    sns.set(style="darkgrid")
    
    # 创建图形
    fig, axes = plt.subplots(4, 1, figsize=(12, 20))
    
    # 1. 价格走势图
    axes[0].plot(merged_df['datetime'], merged_df['close_wti'], label='WTI Crude Oil')
    axes[0].plot(merged_df['datetime'], merged_df['close_brent'], label='Brent Crude Oil')
    axes[0].set_title('WTI vs Brent Crude Oil Price')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend()
    
    # 2. 价差走势图
    axes[1].plot(merged_df['datetime'], merged_df['spread'], color='green')
    axes[1].set_title('Spread Between Brent and WTI (Brent - WTI)')
    axes[1].set_ylabel('Spread (USD)')
    
    # 添加均值线
    mean_spread = merged_df['spread'].mean()
    axes[1].axhline(y=mean_spread, color='r', linestyle='--', label=f'Mean Spread: {mean_spread:.2f}')
    axes[1].legend()
    
    # 3. 价差分布直方图
    sns.histplot(merged_df['spread'], kde=True, ax=axes[2])
    axes[2].set_title('Spread Distribution')
    axes[2].set_xlabel('Spread (USD)')
    axes[2].set_ylabel('Frequency')
    
    # 4. 滚动相关系数图
    axes[3].plot(merged_df['datetime'], merged_df['rolling_corr'], color='purple')
    axes[3].set_title(f'Rolling Correlation ({window_size} data points window)')
    axes[3].set_ylabel('Correlation Coefficient')
    axes[3].set_ylim(-1.1, 1.1)  # 相关系数范围从-1到1
    
    # 添加相关系数均值线
    mean_corr = merged_df['rolling_corr'].mean()
    axes[3].axhline(y=mean_corr, color='r', linestyle='--', label=f'Mean Correlation: {mean_corr:.4f}')
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[3].legend()
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('oil_spread_analysis.png')
    print("Chart saved as 'oil_spread_analysis.png'")
    
    # 保存结果数据
    output_file = 'oil_spread_data.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"Spread data saved as '{output_file}'")

if __name__ == "__main__":
    analyze_oil_spread()