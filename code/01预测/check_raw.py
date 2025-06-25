import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """加载CSV数据"""
    try:
        df = pd.read_csv(file_path)
        print(f"数据加载成功，共{len(df)}行，{len(df.columns)}列")
        return df
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None

def basic_info(df):
    """基本信息检查"""
    print("\n=== 数据基本信息 ===")
    print(f"数据形状: {df.shape}")
    print(f"缺失值情况:")
    print(df.isnull().sum())
    print(f"\n数据类型:")
    print(df.dtypes)
    print(f"\n前5行数据:")
    print(df.head())

def detect_outliers(df, method='iqr'):
    """异常值检测"""
    print(f"\n=== 异常值检测 ({method}方法) ===")
    
    # 选择数值型列
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers_info = {}
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[col]))
            outliers = df[z_scores > 3]
        
        outliers_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'outliers': outliers
        }
        
        print(f"{col}: {len(outliers)}个异常值 ({len(outliers)/len(df)*100:.2f}%)")
    
    return outliers_info

def distribution_analysis(df):
    """分布分析"""
    print("\n=== 分布统计分析 ===")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # 描述性统计
    print("\n描述性统计:")
    print(df[numeric_cols].describe())
    
    # 偏度和峰度
    print("\n偏度和峰度分析:")
    for col in numeric_cols:
        skewness = stats.skew(df[col])
        kurtosis = stats.kurtosis(df[col])
        print(f"{col}: 偏度={skewness:.3f}, 峰度={kurtosis:.3f}")
        
        # 偏度解释
        if abs(skewness) < 0.5:
            skew_desc = "近似对称"
        elif abs(skewness) < 1:
            skew_desc = "轻度偏斜"
        else:
            skew_desc = "高度偏斜"
        
        # 峰度解释
        if abs(kurtosis) < 0.5:
            kurt_desc = "接近正态"
        elif kurtosis > 0.5:
            kurt_desc = "尖峰分布"
        else:
            kurt_desc = "平峰分布"
        
        print(f"  → {skew_desc}, {kurt_desc}")

def normality_test(df):
    """正态性检验"""
    print("\n=== 正态性检验 ===")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Shapiro-Wilk检验 (样本量<5000时使用)
        if len(df) <= 5000:
            stat, p_value = stats.shapiro(df[col])
            test_name = "Shapiro-Wilk"
        else:
            # Kolmogorov-Smirnov检验 (大样本)
            stat, p_value = stats.kstest(df[col], 'norm', args=(df[col].mean(), df[col].std()))
            test_name = "Kolmogorov-Smirnov"
        
        is_normal = p_value > 0.05
        print(f"{col}: {test_name}检验 p值={p_value:.6f} → {'符合正态分布' if is_normal else '不符合正态分布'}")

def plot_distributions(df, save_path=None, sample_size=10000):
    """绘制分布图"""
    print("\n=== 生成分布图 ===")
    
    # 如果数据量大于sample_size，进行抽样
    if len(df) > sample_size:
        df_plot = df.sample(n=sample_size, random_state=42)
        print(f"数据量较大，抽样{sample_size}条进行可视化")
    else:
        df_plot = df
    
    numeric_cols = df_plot.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    
    if n_cols == 0:
        print("没有数值型列可以绘图")
        return
    
    # 计算子图布局
    n_rows = (n_cols + 2) // 3
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numeric_cols):
        row = i // 3
        col_idx = i % 3
        
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        
        # 直方图 + 核密度估计
        sns.histplot(data=df_plot, x=col, kde=True, ax=ax)
        ax.set_title(f'{col} 分布图')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(numeric_cols), n_rows * 3):
        row = i // 3
        col_idx = i % 3
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布图已保存到: {save_path}")
    
    plt.show()

def plot_boxplots(df, save_path=None, sample_size=10000):
    """绘制箱线图"""
    print("\n=== 生成箱线图 ===")
    
    # 如果数据量大于sample_size，进行抽样
    if len(df) > sample_size:
        df_plot = df.sample(n=sample_size, random_state=42)
        print(f"数据量较大，抽样{sample_size}条进行可视化")
    else:
        df_plot = df
    
    numeric_cols = df_plot.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    
    if n_cols == 0:
        print("没有数值型列可以绘图")
        return
    
    # 计算子图布局
    n_rows = (n_cols + 2) // 3
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numeric_cols):
        row = i // 3
        col_idx = i % 3
        
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        
        # 箱线图
        sns.boxplot(data=df_plot, y=col, ax=ax)
        ax.set_title(f'{col} 箱线图')
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(numeric_cols), n_rows * 3):
        row = i // 3
        col_idx = i % 3
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"箱线图已保存到: {save_path}")
    
    plt.show()

def correlation_analysis(df):
    """相关性分析"""
    print("\n=== 相关性分析 ===")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        print("数值型列少于2个，无法进行相关性分析")
        return
    
    # 计算相关系数矩阵
    corr_matrix = df[numeric_cols].corr()
    
    # 找出高相关性的变量对
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:  # 高相关性阈值
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_val
                ))
    
    print("高相关性变量对 (|r| > 0.7):")
    for var1, var2, corr in high_corr_pairs:
        print(f"  {var1} vs {var2}: r = {corr:.3f}")
    
    # 绘制相关性热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('变量相关性热图')
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    # 文件路径
    file_path = r"E:\github_projects\math_modeling_playground\playground\day4\02重命名.csv"
    
    print("开始数据分布异常检查...")
    
    # 加载数据
    df = load_data(file_path)
    if df is None:
        return
    
    # 基本信息检查
    basic_info(df)
    
    # 异常值检测
    outliers_iqr = detect_outliers(df, method='iqr')
    outliers_zscore = detect_outliers(df, method='zscore')
    
    # 分布分析
    distribution_analysis(df)
    
    # 正态性检验
    normality_test(df)
    
    # 相关性分析
    correlation_analysis(df)
    
    # 绘制图形
    plot_distributions(df, "distribution_plots.png")
    plot_boxplots(df, "boxplots.png")
    
    print("\n=== 检查完成 ===")
    print("建议关注:")
    print("1. 异常值比例较高的变量")
    print("2. 高度偏斜的分布")
    print("3. 不符合正态分布的变量")
    print("4. 高相关性的变量对")

if __name__ == "__main__":
    main()