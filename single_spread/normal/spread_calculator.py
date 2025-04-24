import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import os

# 设置matplotlib使用不需要中文支持的字体
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

def parse_formula(formula_str):
    """解析用户输入的价差公式，返回公式中的变量和操作"""
    # 提取公式中的变量（大写字母）
    variables = set(re.findall(r'[A-Z]', formula_str))
    return variables, formula_str

def detect_time_columns(df):
    """智能检测时间相关列"""
    date_cols = []
    time_cols = []
    datetime_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        if 'date' in col_lower and 'time' not in col_lower:
            date_cols.append(col)
        elif 'time' in col_lower and 'date' not in col_lower:
            time_cols.append(col)
        elif 'datetime' in col_lower or ('date' in col_lower and 'time' in col_lower):
            datetime_cols.append(col)
    
    return date_cols, time_cols, datetime_cols

def detect_close_column(df):
    """智能检测收盘价列"""
    for col in df.columns:
        col_lower = col.lower()
        if 'close' in col_lower or 'clos' in col_lower:
            return col
    return None

def resolve_file_path(file_path):
    """解析文件路径，尝试找到文件的实际位置"""
    # 如果是绝对路径，直接返回
    if os.path.isabs(file_path) and os.path.exists(file_path):
        return file_path
        
    # 尝试在当前目录查找文件
    if os.path.exists(file_path):
        return file_path
        
    # 尝试在脚本所在目录查找文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir_path = os.path.join(script_dir, file_path)
    if os.path.exists(script_dir_path):
        print(f"在脚本目录找到文件: {script_dir_path}")
        return script_dir_path
        
    # 尝试在上级目录查找文件
    parent_dir = os.path.dirname(script_dir)
    parent_dir_path = os.path.join(parent_dir, file_path)
    if os.path.exists(parent_dir_path):
        print(f"在上级目录找到文件: {parent_dir_path}")
        return parent_dir_path
    
    # 如果以上都失败，返回原始路径
    print(f"无法找到文件: {file_path}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本目录: {script_dir}")
    
    # 列出当前目录中的文件，帮助用户查看可用文件
    print("\n当前目录下可用的文件:")
    try:
        files = os.listdir(os.getcwd())
        for f in files:
            if os.path.isfile(f):
                print(f"- {f}")
    except Exception as e:
        print(f"无法列出当前目录文件: {str(e)}")
        
    # 列出脚本目录中的文件
    print("\n脚本目录下可用的文件:")
    try:
        files = os.listdir(script_dir)
        for f in files:
            if os.path.isfile(os.path.join(script_dir, f)):
                print(f"- {f}")
    except Exception as e:
        print(f"无法列出脚本目录文件: {str(e)}")
    
    return file_path

def load_market_data(file_path):
    """加载市场数据文件，智能识别格式和列"""
    print(f"尝试读取文件: {file_path}")
    
    # 解析并确认文件路径
    resolved_path = resolve_file_path(file_path)
    print(f"解析后的文件路径: {resolved_path}")
    
    file_ext = os.path.splitext(resolved_path)[1].lower()
    
    # 初始化df为None，用于错误检查
    df = None
    
    if file_ext == '.csv':
        # 尝试不同的分隔符
        for sep in [',', '\t', ';']:
            try:
                print(f"尝试使用分隔符: '{sep}'")
                temp_df = pd.read_csv(resolved_path, sep=sep)
                if len(temp_df.columns) > 1:  # 成功解析为多列
                    df = temp_df
                    print(f"成功读取CSV文件，检测到{len(df.columns)}列")
                    break
            except Exception as e:
                print(f"使用分隔符'{sep}'读取失败: {str(e)}")
                continue
    elif file_ext == '.feather':
        try:
            df = pd.read_feather(resolved_path)
            print(f"成功读取Feather文件，检测到{len(df.columns)}列")
        except Exception as e:
            print(f"读取Feather文件失败: {str(e)}")
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    # 检查df是否成功加载
    if df is None or len(df) == 0:
        raise ValueError(f"无法读取文件: {resolved_path}，请检查文件路径和格式是否正确")
    
    # 输出前几行数据供参考
    print("文件前5行数据:")
    print(df.head())
    
    # 输出所有列名供参考
    print(f"文件列名: {', '.join(df.columns)}")
    
    # 智能检测时间和收盘价列
    date_cols, time_cols, datetime_cols = detect_time_columns(df)
    print(f"检测到的日期列: {date_cols}")
    print(f"检测到的时间列: {time_cols}")
    print(f"检测到的日期时间列: {datetime_cols}")
    
    close_col = detect_close_column(df)
    print(f"检测到的收盘价列: {close_col}")
    
    if not close_col:
        # 尝试查找包含数字的列作为收盘价
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            close_col = numeric_cols[-1]  # 使用最后一个数值列
            print(f"未找到明确的收盘价列，使用数值列: {close_col}")
        else:
            raise ValueError(f"无法在{resolved_path}中找到收盘价列")
    
    # 处理时间列
    if datetime_cols:  # 已有datetime列
        df['datetime'] = pd.to_datetime(df[datetime_cols[0]], errors='coerce')
    elif date_cols and time_cols:  # 有单独的日期和时间列
        try:
            df['datetime'] = pd.to_datetime(df[date_cols[0]] + ' ' + df[time_cols[0]], errors='coerce')
        except:
            # 尝试其他格式
            try:
                df['datetime'] = pd.to_datetime(df[date_cols[0]])
            except:
                raise ValueError(f"无法解析日期时间格式: {date_cols[0]}和{time_cols[0]}")
    elif date_cols:  # 只有日期列
        df['datetime'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    else:
        # 尝试使用索引作为日期时间
        try:
            df['datetime'] = pd.to_datetime(df.index)
        except:
            raise ValueError(f"无法在{resolved_path}中找到时间列")
    
    # 检查datetime列是否有效
    if df['datetime'].isna().all():
        raise ValueError(f"日期时间解析失败，所有值均为NaN")
    
    # 提取需要的列
    result_df = df[['datetime', close_col]].copy()
    result_df.rename(columns={close_col: 'close'}, inplace=True)
    
    # 移除日期时间为NaN的行
    result_df = result_df.dropna(subset=['datetime'])
    print(f"成功加载数据，共{len(result_df)}行")
    
    return result_df

def resample_data(df, freq):
    """根据指定频率重采样数据"""
    print(f"重采样数据到{freq}频率")
    df = df.set_index('datetime')
    resampled = df.resample(freq).last().dropna()
    print(f"重采样后数据行数: {len(resampled)}")
    return resampled.reset_index()

def align_timezone(df, hours_offset):
    """根据时区偏移调整时间"""
    if hours_offset == 0:
        return df
    
    print(f"应用时区偏移: {hours_offset}小时")
    df['datetime'] = df['datetime'] + timedelta(hours=hours_offset)
    return df

def calculate_spread(data_dict, formula):
    """根据公式计算价差"""
    # 创建一个本地命名空间，包含所有变量
    local_vars = {}
    for key, df in data_dict.items():
        local_vars[key] = df['close'].values
    
    # 使用eval计算公式结果
    print(f"使用公式计算价差: {formula}")
    result = eval(formula, {"__builtins__": {}}, local_vars)
    return result

def find_common_timeframe(data_dict):
    """找出所有数据集共有的时间点"""
    if not data_dict:
        return []
        
    common_times = set(data_dict[list(data_dict.keys())[0]]['datetime'])
    
    for key in data_dict:
        current_times = set(data_dict[key]['datetime'])
        common_times = common_times.intersection(current_times)
    
    return sorted(list(common_times))

def main():
    print("=" * 50)
    print("价差计算与分析工具")
    print("=" * 50)
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")
    
    # 步骤1: 输入价差公式
    formula_str = input("请输入价差公式 (例如: A-B, A-2.5*B, 2*B-A-C): ").strip()
    variables, formula = parse_formula(formula_str)
    
    print(f"\n检测到公式中的变量: {', '.join(sorted(variables))}")
    
    # 步骤2: 输入市场数据文件并确认时区
    data_files = {}
    time_offsets = {}
    data_dict = {}
    
    # 显示可用文件供参考
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("\n当前目录下可用的文件:")
    try:
        files = [f for f in os.listdir(os.getcwd()) if os.path.isfile(f)]
        for f in files:
            print(f"- {f}")
    except Exception as e:
        print(f"无法列出当前目录文件: {str(e)}")
    
    print("\n脚本目录下可用的文件:")
    try:
        files = [f for f in os.listdir(script_dir) if os.path.isfile(os.path.join(script_dir, f))]
        for f in files:
            print(f"- {f}")
    except Exception as e:
        print(f"无法列出脚本目录文件: {str(e)}")
    
    for var in sorted(variables):
        file_path = input(f"\n请输入变量 {var} 的市场数据文件路径: ").strip()
        data_files[var] = file_path
        
        tz_offset = input(f"请输入 {var} 的时区偏移量(相对于UTC，如东八区为8，西五区为-5): ").strip()
        try:
            time_offsets[var] = int(tz_offset)
        except ValueError:
            print(f"无效的时区偏移值: {tz_offset}，将使用默认值0")
            time_offsets[var] = 0
    
    # 步骤3: 设置时间精度
    freq_input = input("\n请输入需要的时间精度 (例如: 1min, 5min, 15min, 1h, 1d): ").strip()
    if not freq_input:
        freq_input = "15min"
        print(f"使用默认时间精度: {freq_input}")
    
    # 步骤4: 设置时间范围
    start_date = input("\n请输入起始日期 (格式: YYYY-MM-DD): ").strip()
    end_date = input("请输入结束日期 (格式: YYYY-MM-DD): ").strip()
    
    # 检查日期格式
    try:
        if start_date:
            pd.to_datetime(start_date)
        if end_date:
            pd.to_datetime(end_date)
    except:
        print("日期格式无效，将使用所有可用数据")
        start_date = ""
        end_date = ""
    
    # 加载和处理数据
    print("\n正在加载和处理数据...")
    
    try:
        for var in sorted(variables):
            print(f"\n处理变量 {var} 的数据: {data_files[var]}")
            
            try:
                df = load_market_data(data_files[var])
                
                # 应用时区调整
                df = align_timezone(df, time_offsets[var])
                
                # 过滤时间范围
                if start_date and end_date:
                    original_len = len(df)
                    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]
                    print(f"时间范围过滤: {original_len} -> {len(df)}行")
                    
                    if len(df) == 0:
                        print(f"警告: 在指定时间范围内没有数据")
                        continue
                
                # 重采样数据
                df = resample_data(df, freq_input)
                
                data_dict[var] = df
            except Exception as e:
                print(f"处理变量 {var} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if not data_dict:
            print("没有成功加载任何数据，无法继续分析")
            return
            
        # 找出共同的时间点
        common_times = find_common_timeframe(data_dict)
        print(f"找到 {len(common_times)} 个共同的时间点")
        
        if len(common_times) == 0:
            print("没有找到共同的时间点，无法计算价差")
            return
            
        # 对齐所有数据到共同时间点
        aligned_data = {}
        for var in variables:
            if var in data_dict:
                df = data_dict[var]
                aligned_df = df[df['datetime'].isin(common_times)].copy()
                aligned_data[var] = aligned_df
        
        # 计算价差
        if len(common_times) > 0 and all(var in aligned_data for var in variables):
            result_df = pd.DataFrame({'datetime': common_times})
            for var in variables:
                if var in aligned_data:
                    temp_df = aligned_data[var].set_index('datetime')
                    result_df = result_df.merge(temp_df, on='datetime', how='left', suffixes=('', f'_{var}'))
                    result_df.rename(columns={'close': f'close_{var}'}, inplace=True)
            
            # 计算价差
            try:
                spreads = calculate_spread(aligned_data, formula)
                result_df['spread'] = spreads
                
                # 基本统计分析
                print("\n价差统计分析:")
                spread_stats = {
                    "平均价差": np.mean(spreads),
                    "最大价差": np.max(spreads),
                    "最小价差": np.min(spreads),
                    "标准差": np.std(spreads),
                    "中位数": np.median(spreads)
                }
                
                for stat, value in spread_stats.items():
                    print(f"{stat}: {value:.4f}")
                
                # 数据可视化
                print("\n创建可视化图表...")
                
                # 设置图表风格
                sns.set(style="darkgrid")
                
                # 创建图形
                fig, axes = plt.subplots(3, 1, figsize=(12, 15))
                
                # 1. 原始价格图
                for var in sorted(variables):
                    if var in aligned_data:
                        axes[0].plot(result_df['datetime'], result_df[f'close_{var}'], label=var)
                
                axes[0].set_title('Original Price Data')
                axes[0].set_ylabel('Price')
                axes[0].legend()
                
                # 2. 价差走势图
                axes[1].plot(result_df['datetime'], result_df['spread'], color='green')
                axes[1].set_title(f'Spread Trend ({formula_str})')
                axes[1].set_ylabel('Spread')
                
                # 添加均值线
                mean_spread = np.mean(spreads)
                axes[1].axhline(y=mean_spread, color='r', linestyle='--', label=f'Mean Spread: {mean_spread:.4f}')
                axes[1].legend()
                
                # 3. 价差分布直方图
                sns.histplot(result_df['spread'], kde=True, ax=axes[2])
                axes[2].set_title('Spread Distribution')
                axes[2].set_xlabel('Spread')
                axes[2].set_ylabel('Frequency')
                
                # 保存图表
                plt.tight_layout()
                output_img = os.path.join(script_dir, 'spread_analysis.png')
                plt.savefig(output_img)
                print(f"图表已保存为 '{output_img}'")
                
                # 保存结果数据
                output_file = os.path.join(script_dir, 'spread_data.csv')
                result_df.to_csv(output_file, index=False)
                print(f"价差数据已保存为 '{output_file}'")
            except Exception as e:
                print(f"计算价差时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            missing_vars = [var for var in variables if var not in aligned_data]
            if missing_vars:
                print(f"变量 {', '.join(missing_vars)} 没有有效数据，无法计算价差")
            else:
                print("没有找到共同的时间点，无法计算价差")
    except Exception as e:
        print(f"处理数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc() 