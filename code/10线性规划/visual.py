import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_results(csv_path):
    """
    加载优化结果CSV文件，并生成可视化对比图表。
    """
    # --- 1. 加载数据与准备 ---
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误：找不到结果文件 '{csv_path}'。请先运行主优化脚本。")
        return

    # 筛选出成功优化的数据行
    df = df[df['status'] == 'Optimal'].copy()
    if df.empty:
        print("文件中没有状态为 'Optimal' 的成功优化记录，无法生成图表。")
        return

    # --- 2. 设置绘图风格与中文字体 ---
    # 这对于在图表中正确显示中文至关重要
    sns.set_theme(style="whitegrid")
    try:
        # 优先使用黑体，可替换为您系统中已安装的其他中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    except Exception as e:
        print(f"警告：设置中文字体失败，图表中的中文可能显示为方框。错误: {e}")
        print("请尝试将 plt.rcParams['font.sans-serif'] = ['SimHei'] 中的 'SimHei' 替换为您电脑上已安装的字体，如 'Microsoft YaHei'。")


    # --- 3. 计算百分比变化并打印摘要 ---
    df['cong_pct_change'] = ((df['cong_index_new'] - df['cong_index_orig']) / df['cong_index_orig']) * 100
    df['safety_pct_change'] = ((df['safety_index_new'] - df['safety_index_orig']) / df['safety_index_orig']) * 100
    df['co2_pct_change'] = ((df['co2_new'] - df['co2_orig']) / df['co2_orig']) * 100
    df['passenger_pct_change'] = ((df['passenger_new'] - df['passenger_orig']) / df['passenger_orig']) * 100
    df['electric_pct_change'] = ((df['optimal_ebike_count'] - df['ebike_count_orig']) / df['ebike_count_orig']) * 100
    
    print("--- 平均优化效果摘要 ---")
    print(f"优化道路数量: {len(df)} 条")
    print(f"拥挤指数平均降低: {-df['cong_pct_change'].mean():.2f}%")
    print(f"安全指数平均提升: {df['safety_pct_change'].mean():.2f}%")
    print(f"CO2排放量平均变化: {df['co2_pct_change'].mean():.2f}%")
    print(f"载客量平均变化: {df['passenger_pct_change'].mean():.2f}%")
    print(f"电动车数量平均变化: {df['electric_pct_change'].mean():.2f}%")
    print("-" * 25 + "\n")

    # --- 4. 绘制图表: 优化效果汇总条形图 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('优化效果总体统计', fontsize=16, y=0.95)
    
    # 平均变化条形图
    metrics = ['拥挤指数', '安全指数', 'CO₂排放', '载客量', '电动车数量']
    mean_changes = [df['cong_pct_change'].mean(), df['safety_pct_change'].mean(), df['co2_pct_change'].mean(), df['passenger_pct_change'].mean(), df['electric_pct_change'].mean()]
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral']
    
    bars = ax1.bar(metrics, mean_changes, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
    ax1.set_title('各指标平均变化', fontsize=14, pad=20)
    ax1.set_ylabel('平均变化百分比 (%)')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 在条形图上显示数值
    for bar, value in zip(bars, mean_changes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -3),
                f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 改变方向分布堆积条形图
    change_columns = ['cong_pct_change', 'safety_pct_change', 'co2_pct_change', 'passenger_pct_change', 'electric_pct_change']
    
    # 计算各指标的正向、负向、无变化数量
    positive_counts = []
    negative_counts = []
    no_change_counts = []
    
    for col in change_columns:
        positive = (df[col] > 0).sum()
        negative = (df[col] < 0).sum()
        no_change = (df[col] == 0).sum()
        
        positive_counts.append(positive)
        negative_counts.append(negative)
        no_change_counts.append(no_change)
    
    # 创建堆积条形图
    width = 0.6
    x = np.arange(len(metrics))
    
    p1 = ax2.bar(x, positive_counts, width, label='正向变化', color='lightgreen', alpha=0.8)
    p2 = ax2.bar(x, negative_counts, width, bottom=positive_counts, label='负向变化', color='lightcoral', alpha=0.8)
    p3 = ax2.bar(x, no_change_counts, width, bottom=np.array(positive_counts) + np.array(negative_counts), 
                 label='无变化', color='lightgray', alpha=0.8)
    
    ax2.set_title('各指标变化方向分布', fontsize=14, pad=20)
    ax2.set_ylabel('道路数量')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 在堆积条形图上显示数值
    for i, (pos, neg, no_change) in enumerate(zip(positive_counts, negative_counts, no_change_counts)):
        if pos > 0:
            ax2.text(i, pos/2, str(pos), ha='center', va='center', fontweight='bold', color='white')
        if neg > 0:
            ax2.text(i, pos + neg/2, str(neg), ha='center', va='center', fontweight='bold', color='white')
        if no_change > 0:
            ax2.text(i, pos + neg + no_change/2, str(no_change), ha='center', va='center', fontweight='bold', color='black')
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

if __name__ == '__main__':
    # --- 请在这里指定上一步生成的CSV文件路径 ---
    RESULTS_CSV_PATH = r'optimization_results_高拥堵_均衡模式.csv'
    # -----------------------------------------
    
    visualize_results(RESULTS_CSV_PATH)