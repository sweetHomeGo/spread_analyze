import pandas as pd
import pyarrow.feather as feather
from pathlib import Path
import glob

def merge_contract_data(input_dir, output_file):
    """合并合约数据为宽表格式"""
    merged_df = pd.DataFrame()
    
    for file_path in glob.glob(f"{input_dir}/*.csv"):
        try:
            contract = Path(file_path).stem.split('_')[0]  # 假设文件名格式为I1501.csv
            
            # 读取单个合约数据
            df = pd.read_csv(
                file_path,
                usecols=['datetime', 'close'],
                parse_dates=['datetime'],
                dtype={'close': 'float32'}
            ).rename(columns={'datetime': 'timestamp', 'close': contract})
            
            # 检查并填充合约数据中的NaN值
            if df[contract].isna().any():
                print(f"合约 {contract} 存在NaN值，使用前值填充")
                df[contract] = df[contract].fillna(method='ffill')
                # 如果开头有NaN值，使用后值填充
                df[contract] = df[contract].fillna(method='bfill')
            
            # 按时间戳合并到主表
            if merged_df.empty:
                merged_df = df
            else:
                merged_df = pd.merge(
                    merged_df, 
                    df,
                    on='timestamp',
                    how='outer',  # 外连接保留所有时间点
                    suffixes=('', '_dup')
                )
                # 处理可能的列重复
                merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            continue

    # 按时间排序并去重
    merged_df = merged_df.sort_values('timestamp').drop_duplicates('timestamp')
    
    # 保存为feather格式
    merged_df.reset_index(drop=True).to_feather(output_file)
    print(f"生成宽表数据，包含 {len(merged_df.columns)-1} 个合约，总时间点：{len(merged_df)}")

# 使用示例
if __name__ == "__main__":
    merge_contract_data(
        input_dir="./I",  # 原始数据目录
        output_file="./merged.feather"  # 输出文件路径
    )
