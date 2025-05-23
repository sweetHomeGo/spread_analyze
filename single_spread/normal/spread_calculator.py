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

def resolve_file_path(file_path, data_dir):
    """解析文件路径，优先在data文件夹中查找文件"""
    # 如果是绝对路径，直接返回
    if os.path.isabs(file_path) and os.path.exists(file_path):
        return file_path
    
    # 首先在data目录查找文件
    data_dir_path = os.path.join(data_dir, file_path)
    if os.path.exists(data_dir_path):
        print(f"在data目录找到文件: {data_dir_path}")
        return data_dir_path
        
    # 尝试在当前目录查找文件
    if os.path.exists(file_path):
        return file_path
        
    # 尝试在脚本所在目录查找文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir_path = os.path.join(script_dir, file_path)
    if os.path.exists(script_dir_path):
        print(f"在脚本目录找到文件: {script_dir_path}")
        return script_dir_path
    
    # 如果以上都失败，返回原始路径
    print(f"无法找到文件: {file_path}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"data目录: {data_dir}")
    
    # 列出data目录中的文件，帮助用户查看可用文件
    print("\ndata目录下可用的文件:")
    try:
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        for f in files:
            print(f"- {f}")
    except Exception as e:
        print(f"无法列出data目录文件: {str(e)}")
    
    return file_path

def load_market_data(file_path, data_dir):
    """加载市场数据文件，智能识别格式和列"""
    print(f"尝试读取文件: {file_path}")
    
    # 解析并确认文件路径
    resolved_path = resolve_file_path(file_path, data_dir)
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

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")
    return directory

def calculate_rolling_correlation(df, variables, window_size=20):
    """计算两个变量之间的滚动相关系数"""
    if len(variables) < 2:
        return None
    
    # 取前两个变量计算相关系数
    var1, var2 = sorted(variables)[:2]
    col1 = f'close_{var1}'
    col2 = f'close_{var2}'
    
    # 确保这些列存在
    if col1 not in df.columns or col2 not in df.columns:
        print(f"警告：找不到列 {col1} 或 {col2}，无法计算相关系数")
        print(f"可用列：{', '.join(df.columns)}")
        return None
    
    # 确保数据按时间排序
    df = df.sort_values('datetime')
    
    # 检查是否有足够的数据点
    if len(df) < window_size:
        print(f"警告：数据点数量({len(df)})小于窗口大小({window_size})，无法计算滚动相关系数")
        return None
    
    # 检查数据中是否存在NaN值
    nan_count1 = df[col1].isna().sum()
    nan_count2 = df[col2].isna().sum()
    if nan_count1 > 0 or nan_count2 > 0:
        print(f"警告：数据中存在NaN值（{col1}: {nan_count1}, {col2}: {nan_count2}），这可能影响相关系数计算")
    
    # 计算滚动相关系数，处理NaN值
    rolling_corr = df[col1].rolling(window=window_size, min_periods=int(window_size/2)).corr(df[col2])
    
    return rolling_corr

def main():
    print("=" * 50)
    print("价差计算与分析工具")
    print("=" * 50)
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")
    
    # 创建数据、价差结果和图表的输出目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = ensure_dir(os.path.join(script_dir, "data"))
    charts_dir = ensure_dir(os.path.join(script_dir, "charts"))
    spreads_dir = ensure_dir(os.path.join(script_dir, "spreads"))  # 新增spreads目录
    
    # 步骤1: 输入价差公式
    formula_str = input("请输入价差公式 (例如: A-B, A-2.5*B, 2*B-A-C): ").strip()
    variables, formula = parse_formula(formula_str)
    
    print(f"\n检测到公式中的变量: {', '.join(sorted(variables))}")
    
    # 步骤2: 扫描数据文件并通过编号选择
    data_files = {}
    time_offsets = {}
    data_dict = {}
    
    # 扫描data目录下可用的文件并编号列出
    available_files = []
    try:
        files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        if files:
            print("\ndata目录下可用的文件:")
            for i, f in enumerate(sorted(files), 1):
                print(f"{i}. {f}")
                available_files.append(f)
        else:
            print("data目录中没有文件")
            return
    except Exception as e:
        print(f"无法列出data目录文件: {str(e)}")
        return
    
    # 通过编号选择文件
    for var in sorted(variables):
        valid_selection = False
        while not valid_selection:
            try:
                file_index = input(f"\n请输入变量 {var} 对应的文件编号 (1-{len(available_files)}): ").strip()
                file_index = int(file_index)
                if 1 <= file_index <= len(available_files):
                    selected_file = available_files[file_index-1]
                    data_files[var] = os.path.join(data_dir, selected_file)
                    print(f"已选择: {selected_file}")
                    valid_selection = True
                else:
                    print(f"请输入有效的编号 (1-{len(available_files)})")
            except ValueError:
                print("请输入有效的数字编号")
        
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
    
    # 设置滚动窗口大小
    window_size_input = input("\n请输入滚动相关性窗口大小 (默认值: 20): ").strip()
    try:
        window_size = int(window_size_input) if window_size_input else 20
    except ValueError:
        window_size = 20
        print(f"无效的窗口大小，使用默认值: {window_size}")
    
    # 步骤4: 设置时间范围
    start_date = input("\n请输入起始日期 (格式: YYYY-MM-DD): ").strip()
    end_date = input("请输入结束日期 (格式: YYYY-MM-DD): ").strip()
    
    # 创建文件名基础
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_base_name = f"spread_{formula_str.replace(' ', '').replace('*', 'x').replace('/', 'div')}_{timestamp}"
    
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
            print(f"\n处理变量 {var} 的数据: {os.path.basename(data_files[var])}")
            
            try:
                # 直接使用完整路径加载数据，不需要再次搜索
                file_path = data_files[var]
                print(f"读取文件: {file_path}")
                
                file_ext = os.path.splitext(file_path)[1].lower()
                # 初始化df为None，用于错误检查
                df = None
                
                if file_ext == '.csv':
                    # 尝试不同的分隔符
                    for sep in [',', '\t', ';']:
                        try:
                            print(f"尝试使用分隔符: '{sep}'")
                            temp_df = pd.read_csv(file_path, sep=sep)
                            if len(temp_df.columns) > 1:  # 成功解析为多列
                                df = temp_df
                                print(f"成功读取CSV文件，检测到{len(df.columns)}列")
                                break
                        except Exception as e:
                            print(f"使用分隔符'{sep}'读取失败: {str(e)}")
                            continue
                elif file_ext == '.feather':
                    try:
                        df = pd.read_feather(file_path)
                        print(f"成功读取Feather文件，检测到{len(df.columns)}列")
                    except Exception as e:
                        print(f"读取Feather文件失败: {str(e)}")
                else:
                    raise ValueError(f"不支持的文件格式: {file_ext}")
                
                # 检查df是否成功加载
                if df is None or len(df) == 0:
                    raise ValueError(f"无法读取文件: {file_path}，请检查文件格式是否正确")
                
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
                        raise ValueError(f"无法在{file_path}中找到收盘价列")
                
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
                        raise ValueError(f"无法在{file_path}中找到时间列")
                
                # 检查datetime列是否有效
                if df['datetime'].isna().all():
                    raise ValueError(f"日期时间解析失败，所有值均为NaN")
                
                # 提取需要的列
                result_df = df[['datetime', close_col]].copy()
                result_df.rename(columns={close_col: 'close'}, inplace=True)
                
                # 移除日期时间为NaN的行
                result_df = result_df.dropna(subset=['datetime'])
                print(f"成功加载数据，共{len(result_df)}行")
                
                # 应用时区调整
                df = align_timezone(result_df, time_offsets[var])
                
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
        
        # 添加这段代码来根据用户输入的日期范围过滤数据
        if start_date or end_date:
            print(f"根据用户指定的时间范围过滤数据: {start_date or '最早'} 到 {end_date or '最新'}")
            for var in aligned_data:
                df = aligned_data[var]
                if start_date:
                    start_datetime = pd.to_datetime(start_date)
                    df = df[df['datetime'] >= start_datetime]
                if end_date:
                    end_datetime = pd.to_datetime(end_date)
                    # 将结束日期调整到当天结束
                    end_datetime = end_datetime.replace(hour=23, minute=59, second=59)
                    df = df[df['datetime'] <= end_datetime]
                aligned_data[var] = df
                print(f"变量 {var} 过滤后的数据行数: {len(df)}")
            
            # 重新找出共同的时间点
            common_times = find_common_timeframe(aligned_data)
            print(f"过滤后找到 {len(common_times)} 个共同的时间点")
            
            # 再次对齐数据
            for var in variables:
                if var in aligned_data:
                    df = aligned_data[var]
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
                
                # 计算滚动相关系数（如果有至少两个变量）
                if len(variables) >= 2:
                    rolling_corr = calculate_rolling_correlation(result_df, variables, window_size)
                    if rolling_corr is not None:
                        result_df['rolling_corr'] = rolling_corr
                        print(f"计算了滚动相关系数，窗口大小: {window_size}")
                
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
                
                # 如果有至少两个变量，计算相关系数
                if len(variables) >= 2:
                    var1, var2 = sorted(variables)[:2]
                    col1 = f'close_{var1}'
                    col2 = f'close_{var2}'
                    
                    # 确保这些列存在
                    if col1 in result_df.columns and col2 in result_df.columns:
                        # 使用dropna确保在计算相关系数时排除了NaN值
                        correlation = result_df[[col1, col2]].dropna().corr().iloc[0, 1]
                        print(f"\n{var1}和{var2}的相关系数: {correlation:.4f}")
                    else:
                        print(f"警告：找不到列 {col1} 或 {col2}，无法计算相关系数")
                
                # 保存结果数据到CSV文件 - 改为保存到spreads目录
                csv_filename = f"{file_base_name}.csv"
                output_csv = os.path.join(spreads_dir, csv_filename)  # 修改为保存到spreads目录
                result_df.to_csv(output_csv, index=False)
                print(f"价差数据已保存为: '{output_csv}'")
                
                # 数据可视化
                print("\n创建可视化图表...")
                
                # 设置图表风格
                sns.set(style="darkgrid")
                
                # 创建图形 - 如果有相关系数分析则创建4个子图，否则创建3个
                if len(variables) >= 2 and 'rolling_corr' in result_df.columns:
                    fig, axes = plt.subplots(4, 1, figsize=(12, 20))
                else:
                    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
                
                # 1. 原始价格图 - 使用双Y轴
                if len(variables) >= 2:
                    var1, var2 = sorted(variables)[:2]
                    
                    # 创建主Y轴（左侧）
                    color1 = 'tab:blue'
                    axes[0].set_ylabel(f'{var1} Price', color=color1)
                    axes[0].plot(result_df['datetime'], result_df[f'close_{var1}'], color=color1, label=var1)
                    axes[0].tick_params(axis='y', labelcolor=color1)
                    
                    # 创建次Y轴（右侧）
                    color2 = 'tab:red'
                    ax2 = axes[0].twinx()
                    ax2.set_ylabel(f'{var2} Price', color=color2)
                    ax2.plot(result_df['datetime'], result_df[f'close_{var2}'], color=color2, label=var2)
                    ax2.tick_params(axis='y', labelcolor=color2)
                    
                    # 创建合并的图例
                    lines1, labels1 = axes[0].get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    axes[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                else:
                    # 如果只有一个变量，使用单一Y轴
                    for var in sorted(variables):
                        if var in aligned_data:
                            axes[0].plot(result_df['datetime'], result_df[f'close_{var}'], label=var)
                    axes[0].set_ylabel('Price')
                    axes[0].legend()
                
                axes[0].set_title('Original Price Data')
                
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
                
                # 4. 滚动相关系数图（如果有）
                if len(variables) >= 2 and 'rolling_corr' in result_df.columns:
                    var1, var2 = sorted(variables)[:2]
                    axes[3].plot(result_df['datetime'], result_df['rolling_corr'], color='purple')
                    axes[3].set_title(f'Rolling Correlation between {var1} and {var2} (Window Size: {window_size})')
                    axes[3].set_ylabel('Correlation Coefficient')
                    
                    # 自适应纵坐标范围，但最小要包含-1到1的范围
                    y_min = max(min(result_df['rolling_corr'].min() * 1.1, -1.1), -1.1)
                    y_max = min(max(result_df['rolling_corr'].max() * 1.1, 1.1), 1.1)
                    axes[3].set_ylim(y_min, y_max)
                    
                    # 添加零线和1/-1线
                    axes[3].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                    axes[3].axhline(y=1, color='gray', linestyle='--', alpha=0.3)
                    axes[3].axhline(y=-1, color='gray', linestyle='--', alpha=0.3)
                    
                    # 添加相关性强度区域
                    axes[3].axhspan(0.7, 1, alpha=0.1, color='green', label='Strong Positive')
                    axes[3].axhspan(-1, -0.7, alpha=0.1, color='red', label='Strong Negative')
                    axes[3].legend()
                
                # 保存组合图表
                plt.tight_layout()
                combined_chart_path = os.path.join(charts_dir, f"{file_base_name}_combined.png")
                plt.savefig(combined_chart_path)
                print(f"图表已保存为: '{combined_chart_path}'")
                
                print(f"已保存汇总图表到 '{charts_dir}' 目录")
                
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