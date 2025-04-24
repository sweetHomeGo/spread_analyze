import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import seaborn as sns
from datetime import datetime

class SpreadVisualizer:
    """价差图表生成器"""
    
    def __init__(self, spread_prices_file, spread_list_file=None):
        """初始化可视化器"""
        # 加载价差价格数据
        self.prices_df = pd.read_feather(spread_prices_file)
        print(f"Loaded price data with {len(self.prices_df)} rows and {len(self.prices_df.columns)} columns")
        
        # 加载价差列表文件（如果提供）
        if spread_list_file and Path(spread_list_file).exists():
            self.spreads_df = pd.read_csv(spread_list_file)
            print(f"Loaded {len(self.spreads_df)} spread definitions")
        else:
            self.spreads_df = None
            print("No spread list file provided, will use all available spreads")
        
        # 设置绘图样式
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['font.size'] = 12
    
    def extract_contract_month(self, contract_code):
        """从合约代码中提取月份"""
        import re
        match = re.match(r"^[A-Za-z]+\d{2}(\d{2})$", contract_code)
        if match:
            return int(match.group(1))
        return None
    
    def filter_spreads(self, main_month, spread_type):
        """筛选符合主力月份和价差类型的价差"""
        if self.spreads_df is None:
            return []
            
        # 首先根据价差类型筛选
        type_filtered = self.spreads_df[self.spreads_df['spread_type'] == spread_type]
        
        # 然后根据主力月份筛选
        valid_spreads = []
        
        for _, row in type_filtered.iterrows():
            spread_code = row['spread_code']
            contract_a = row['contract_a']
            
            # 对于不同的价差类型，主力合约可能是A或B
            if spread_type in ['PrevMonth-Main', 'PrevPrevMonth-Main']:
                # 这些类型中，主力合约是B
                main_contract = row['contract_b']
            else:
                # 其他类型中，主力合约是A
                main_contract = contract_a
            
            # 检查主力合约的月份是否匹配
            month = self.extract_contract_month(main_contract)
            
            if month == main_month:
                # 确保价差在价格数据中存在
                if spread_code in self.prices_df.columns:
                    valid_spreads.append(spread_code)
        
        return valid_spreads
    
    def get_spread_types(self):
        """获取所有价差类型"""
        if self.spreads_df is None:
            return []
        return sorted(self.spreads_df['spread_type'].unique())
    
    def plot_simple_spreads(self, spread_codes):
        """使用简单索引绘制价差，并将图片保存在专门的文件夹内"""
        if not spread_codes:
            print("No spreads selected")
            return
        
        # 检查有效的价差
        valid_spreads = [s for s in spread_codes if s in self.prices_df.columns]
        if not valid_spreads:
            print("No valid spreads to plot")
            return
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # 使用不同颜色区分不同价差
        colors = plt.cm.tab10.colors  # 使用tab10调色板
        
        # 为每个价差绘制一条线
        for i, spread in enumerate(valid_spreads):
            # 获取该价差的所有有效数据点
            spread_data = self.prices_df[spread].dropna()
            
            if len(spread_data) == 0:
                print(f"No valid data for {spread}")
                continue
            
            # 创建简单索引
            indices = range(len(spread_data))
            
            # 选择颜色
            color = colors[i % len(colors)]
            
            # 绘制线条 - 使用简单索引作为x轴
            line, = ax.plot(
                indices, 
                spread_data.values,
                color=color,
                linewidth=1.5,
                alpha=0.8,
                label=spread
            )
            
            print(f"Plotted {len(spread_data)} points for {spread}")
        
        # 设置x轴刻度（简单的数字刻度）
        ax.xaxis.set_major_locator(mticker.MaxNLocator(10))
        
        # 添加图表元素
        plt.title("Spread Price Comparison", fontsize=16)
        plt.xlabel("Data Point Index", fontsize=14)
        plt.ylabel("Spread Price", fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        plt.legend(loc='best', fontsize=12)
        
        # 添加水平线表示零点
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # 添加数据统计信息
        stats_lines = []
        for spread in valid_spreads:
            spread_data = self.prices_df[spread].dropna()
            if len(spread_data) > 0:
                stats_lines.append(
                    f"{spread}: Points={len(spread_data):,}, "
                    f"Min={spread_data.min():.2f}, "
                    f"Max={spread_data.max():.2f}, "
                    f"Avg={spread_data.mean():.2f}"
                )
        
        # 添加统计信息文本框
        if stats_lines:
            plt.figtext(0.02, 0.02, "\n".join(stats_lines), fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8))
        
        # 调整布局
        plt.tight_layout()
        
        # 创建图表保存目录
        charts_dir = Path("./spread_charts")
        charts_dir.mkdir(exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        spread_names = "_".join([s.replace('-', '') for s in valid_spreads[:2]])  # 使用前两个价差名称
        if len(valid_spreads) > 2:
            spread_names += f"_and_{len(valid_spreads)-2}_more"
        
        filename = f"{spread_names}_{timestamp}.png"
        output_path = charts_dir / filename
        
        # 保存图表
        plt.savefig(output_path, dpi=300)
        print(f"Chart saved as {output_path}")
        
        # 显示图表
        plt.show()
        
        return fig, ax
    
    def interactive_plot(self):
        """交互式选择并绘制价差图表"""
        if self.spreads_df is None:
            print("No spread list file provided. Please select spreads directly:")
            available_spreads = [col for col in self.prices_df.columns if col != 'timestamp']
            if len(available_spreads) > 20:
                print(f"Found {len(available_spreads)} spreads. Showing first 20:")
                for i, spread in enumerate(available_spreads[:20], 1):
                    print(f"{i}: {spread}")
                print(f"... and {len(available_spreads) - 20} more")
            else:
                for i, spread in enumerate(available_spreads, 1):
                    print(f"{i}: {spread}")
            
            print("\nSelect spreads to plot (comma-separated numbers, e.g., '1,3,5' or 'all' for all):")
            selection = input("> ").strip().lower()
            
            selected_spreads = []
            if selection == 'all':
                selected_spreads = available_spreads
            else:
                try:
                    indices = [int(idx.strip()) for idx in selection.split(',') if idx.strip()]
                    for idx in indices:
                        if 1 <= idx <= len(available_spreads):
                            selected_spreads.append(available_spreads[idx-1])
                        else:
                            print(f"Warning: Index {idx} out of range, ignored.")
                except ValueError:
                    print("Invalid input format. Using first spread as default.")
                    if available_spreads:
                        selected_spreads = [available_spreads[0]]
            
            if not selected_spreads and available_spreads:
                print("No valid spreads selected. Using first spread as default.")
                selected_spreads = [available_spreads[0]]
                
            print(f"\nPlotting {len(selected_spreads)} spreads: {', '.join(selected_spreads)}")
            self.plot_simple_spreads(selected_spreads)
            return
        
        # 1. 选择主力月份
        print("Available main contract months:")
        print("1: January (01)")
        print("5: May (05)")
        print("9: September (09)")
        
        while True:
            try:
                main_month = int(input("\nSelect main contract month (1, 5, or 9): "))
                if main_month not in [1, 5, 9]:
                    print("Invalid choice. Please select 1, 5, or 9.")
                    continue
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # 2. 选择价差类型
        spread_types = self.get_spread_types()
        print("\nAvailable spread types:")
        for i, spread_type in enumerate(spread_types, 1):
            print(f"{i}: {spread_type}")
        
        while True:
            try:
                type_idx = int(input(f"\nSelect spread type (1-{len(spread_types)}): "))
                if type_idx < 1 or type_idx > len(spread_types):
                    print(f"Invalid choice. Please select a number between 1 and {len(spread_types)}.")
                    continue
                selected_type = spread_types[type_idx-1]
                break
            except ValueError:
                print("Please enter a valid number.")
        
        # 3. 筛选符合条件的价差
        filtered_spreads = self.filter_spreads(main_month, selected_type)
        
        if not filtered_spreads:
            print(f"No spreads found for main month {main_month} and type {selected_type}")
            return
        
        # 4. 多选价差
        print("\nAvailable spreads:")
        for i, spread in enumerate(filtered_spreads, 1):
            print(f"{i}: {spread}")
        
        print("\nSelect spreads to plot (comma-separated numbers, e.g., '1,3,5' or 'all' for all):")
        selection = input("> ").strip().lower()
        
        selected_spreads = []
        if selection == 'all':
            selected_spreads = filtered_spreads
        else:
            try:
                # 解析用户输入的编号
                indices = [int(idx.strip()) for idx in selection.split(',') if idx.strip()]
                for idx in indices:
                    if 1 <= idx <= len(filtered_spreads):
                        selected_spreads.append(filtered_spreads[idx-1])
                    else:
                        print(f"Warning: Index {idx} out of range, ignored.")
            except ValueError:
                print("Invalid input format. Using first spread as default.")
                selected_spreads = [filtered_spreads[0]]
        
        if not selected_spreads:
            print("No valid spreads selected. Using first spread as default.")
            selected_spreads = [filtered_spreads[0]]
        
        # 5. 绘制选定的价差
        print(f"\nPlotting {len(selected_spreads)} spreads: {', '.join(selected_spreads)}")
        self.plot_simple_spreads(selected_spreads)

# 使用示例
if __name__ == "__main__":
    # 文件路径
    spread_prices_file = "./spread_prices.feather"
    spread_list_file = "./iron_spreads.csv"
    
    # 检查文件是否存在
    if not Path(spread_prices_file).exists():
        print(f"Error: Spread prices file not found: {spread_prices_file}")
        exit(1)
    
    # 创建可视化器
    visualizer = SpreadVisualizer(
        spread_prices_file=spread_prices_file,
        spread_list_file=spread_list_file if Path(spread_list_file).exists() else None
    )
    
    # 启动交互式绘图
    visualizer.interactive_plot() 