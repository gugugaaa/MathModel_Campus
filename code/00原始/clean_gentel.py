import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TrafficDataCleaner:
    def __init__(self):
        self.cleaning_log = []
        self.original_count = 0
        self.cleaned_count = 0
        
    def log_step(self, step_name, removed_count, remaining_count):
        """记录清洗步骤"""
        self.cleaning_log.append({
            'step': step_name,
            'removed': removed_count,
            'remaining': remaining_count,
            'removal_rate': f"{removed_count/self.original_count*100:.2f}%"
        })
        print(f"✓ {step_name}: 移除 {removed_count:,} 条记录, 剩余 {remaining_count:,} 条 ({removed_count/self.original_count*100:.2f}%)")
    
    def clean_traffic_data(self, df, gentle_mode=False):
        """
        完整的交通数据清洗流程
        
        参数:
        gentle_mode (bool): True为温和模式，False为标准模式
        """
        print("🚗 开始交通数据清洗...")
        if gentle_mode:
            print("🌱 使用温和清洗模式")
        else:
            print("⚡ 使用标准清洗模式")
        print("=" * 60)
        
        # 备份原始数据
        original_df = df.copy()
        self.original_count = len(df)
        print(f"📊 原始数据量: {self.original_count:,} 条记录")
        print()
        
        # 第一阶段：基础清洗
        print("🔧 第一阶段：基础清洗")
        print("-" * 30)
        if gentle_mode:
            df = self._remove_extreme_outliers_gentle(df)
        else:
            df = self._remove_extreme_outliers(df)
        df = self._physical_constraints_check(df)
        print()
        
        # 第二阶段：逻辑一致性清洗
        print("🔧 第二阶段：逻辑一致性清洗")
        print("-" * 30)
        if gentle_mode:
            df = self._traffic_flow_relationship_check_gentle(df)
        else:
            df = self._traffic_flow_relationship_check(df)
        df = self._traffic_theory_constraints(df)
        print()
        
        # 第三阶段：时间连续性清洗
        print("🔧 第三阶段：时间连续性清洗")
        print("-" * 30)
        df = self._remove_insufficient_data_segments(df)
        if gentle_mode:
            df = self._smooth_time_series_outliers_gentle(df)
        else:
            df = self._smooth_time_series_outliers(df)
        print()
        
        # 第四阶段：统计清洗
        print("🔧 第四阶段：统计清洗")
        print("-" * 30)
        if gentle_mode:
            df = self._percentile_truncation_gentle(df)
        else:
            df = self._percentile_truncation(df)
        df = self._zscore_cleaning(df)
        print()
        
        self.cleaned_count = len(df)
        self._print_cleaning_summary(original_df, df)
        
        return df
    
    def _remove_extreme_outliers_gentle(self, df):
        """移除IQR极端异常值（温和版本）"""
        initial_count = len(df)
        
        # 使用更宽松的4倍IQR范围，只移除真正极端的异常值
        combined_mask = True
        for col in ['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 4 * IQR  # 从3倍改为4倍，更宽松
            upper_bound = Q3 + 4 * IQR
            
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            combined_mask = combined_mask & col_mask
        
        df_cleaned = df[combined_mask].copy()
        
        removed_count = initial_count - len(df_cleaned)
        self.log_step("移除IQR极端异常值(温和)", removed_count, len(df_cleaned))

        return df_cleaned
        
    def _remove_extreme_outliers(self, df):
        """移除IQR极端异常值（标准版本）"""
        initial_count = len(df)
        
        # 使用3倍IQR范围
        combined_mask = True
        for col in ['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            combined_mask = combined_mask & col_mask
        
        df_cleaned = df[combined_mask].copy()
        
        removed_count = initial_count - len(df_cleaned)
        self.log_step("移除IQR极端异常值", removed_count, len(df_cleaned))

        return df_cleaned
        
    def _traffic_flow_relationship_check(self, df):
        """交通流基本关系验证: Flow = Speed × Density（标准版本）"""
        initial_count = len(df)
        
        # 计算理论流量
        theoretical_flow = df['SPEED_kph'] * df['DENSITY_vpkm']
        
        # 计算相对误差
        relative_error = np.abs(df['FLOW_vph'] - theoretical_flow) / np.maximum(df['FLOW_vph'], 1)
        
        # 保留误差小于15%的记录
        flow_consistency_mask = relative_error <= 0.15
        
        df_cleaned = df[flow_consistency_mask].copy()
        removed_count = initial_count - len(df_cleaned)
        self.log_step("交通流基本关系验证", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _smooth_time_series_outliers(self, df):
        """时间序列异常平滑处理（标准版本）"""
        initial_count = len(df)
        
        # 按路段和日期排序
        df_sorted = df.sort_values(['ROADSECT_ID', 'DATE', 'TIME']).copy()
        
        # 计算相邻时间点的变化率
        for col in ['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']:
            # 按路段分组计算变化率
            df_sorted[f'{col}_change_rate'] = df_sorted.groupby('ROADSECT_ID')[col].pct_change().abs()
        
        # 标记变化率过大的点
        high_change_mask = (
            (df_sorted['SPEED_kph_change_rate'] > 0.8) |
            (df_sorted['FLOW_vph_change_rate'] > 1.0) |
            (df_sorted['DENSITY_vpkm_change_rate'] > 1.0)
        )
        
        # 移除突变点
        df_cleaned = df_sorted[~high_change_mask.fillna(False)].copy()
        
        # 清理临时列
        change_cols = [col for col in df_cleaned.columns if col.endswith('_change_rate')]
        df_cleaned = df_cleaned.drop(columns=change_cols)
        
        removed_count = initial_count - len(df_cleaned)
        self.log_step("时间序列突变处理", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _percentile_truncation(self, df):
        """分位数截断（标准版本）"""
        initial_count = len(df)
        
        # 使用1%-99%分位数截断
        percentile_mask = True
        for col in ['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']:
            p1 = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            percentile_mask = percentile_mask & (df[col] >= p1) & (df[col] <= p99)
        
        df_cleaned = df[percentile_mask].copy()
        removed_count = initial_count - len(df_cleaned)
        self.log_step("分位数截断(1%-99%)", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _traffic_flow_relationship_check_gentle(self, df):
        """交通流基本关系验证: Flow = Speed × Density（温和版本）"""
        initial_count = len(df)
        
        # 计算理论流量
        theoretical_flow = df['SPEED_kph'] * df['DENSITY_vpkm']
        
        # 计算相对误差，使用更宽松的阈值
        relative_error = np.abs(df['FLOW_vph'] - theoretical_flow) / np.maximum(df['FLOW_vph'], 1)
        
        # 保留误差小于25%的记录（比标准版的15%更宽松）
        flow_consistency_mask = relative_error <= 0.25
        
        df_cleaned = df[flow_consistency_mask].copy()
        removed_count = initial_count - len(df_cleaned)
        self.log_step("交通流基本关系验证(温和)", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _smooth_time_series_outliers_gentle(self, df):
        """时间序列异常平滑处理（温和版本）"""
        initial_count = len(df)
        
        # 按路段和日期排序
        df_sorted = df.sort_values(['ROADSECT_ID', 'DATE', 'TIME']).copy()
        
        # 计算相邻时间点的变化率，使用更宽松的阈值
        for col in ['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']:
            # 按路段分组计算变化率
            df_sorted[f'{col}_change_rate'] = df_sorted.groupby('ROADSECT_ID')[col].pct_change().abs()
        
        # 使用更宽松的变化率阈值，只移除极端突变
        high_change_mask = (
            (df_sorted['SPEED_kph_change_rate'] > 1.5) |  # 从0.8改为1.5
            (df_sorted['FLOW_vph_change_rate'] > 2.0) |   # 从1.0改为2.0
            (df_sorted['DENSITY_vpkm_change_rate'] > 2.0)  # 从1.0改为2.0
        )
        
        # 移除突变点
        df_cleaned = df_sorted[~high_change_mask.fillna(False)].copy()
        
        # 清理临时列
        change_cols = [col for col in df_cleaned.columns if col.endswith('_change_rate')]
        df_cleaned = df_cleaned.drop(columns=change_cols)
        
        removed_count = initial_count - len(df_cleaned)
        self.log_step("时间序列突变处理(温和)", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _percentile_truncation_gentle(self, df):
        """分位数截断（温和版本）"""
        initial_count = len(df)
        
        # 使用0.5%-99.5%分位数截断，比1%-99%更宽松
        percentile_mask = True
        for col in ['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']:
            p05 = df[col].quantile(0.005)  # 0.5%分位数
            p995 = df[col].quantile(0.995)  # 99.5%分位数
            percentile_mask = percentile_mask & (df[col] >= p05) & (df[col] <= p995)
        
        df_cleaned = df[percentile_mask].copy()
        removed_count = initial_count - len(df_cleaned)
        self.log_step("分位数截断(0.5%-99.5%)", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _physical_constraints_check(self, df):
        """物理约束检查"""
        initial_count = len(df)
        
        # 物理合理性检查
        physical_mask = (
            (df['SPEED_kph'] > 0) & (df['SPEED_kph'] <= 120) &  # 速度合理范围
            (df['FLOW_vph'] >= 0) & (df['FLOW_vph'] <= 3000) &  # 流量合理范围
            (df['DENSITY_vpkm'] > 0) & (df['DENSITY_vpkm'] <= 200)  # 密度合理范围
        )
        
        df_cleaned = df[physical_mask].copy()
        removed_count = initial_count - len(df_cleaned)
        self.log_step("物理约束检查", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _traffic_flow_relationship_check(self, df):
        """交通流基本关系验证: Flow = Speed × Density"""
        initial_count = len(df)
        
        # 计算理论流量
        theoretical_flow = df['SPEED_kph'] * df['DENSITY_vpkm']
        
        # 计算相对误差
        relative_error = np.abs(df['FLOW_vph'] - theoretical_flow) / np.maximum(df['FLOW_vph'], 1)
        
        # 保留误差小于15%的记录（比报告中的10%稍宽松）
        flow_consistency_mask = relative_error <= 0.15
        
        df_cleaned = df[flow_consistency_mask].copy()
        removed_count = initial_count - len(df_cleaned)
        self.log_step("交通流基本关系验证", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _traffic_theory_constraints(self, df):
        """交通流理论约束"""
        initial_count = len(df)
        
        # 移除违背交通流理论的记录
        theory_mask = ~(
            (df['DENSITY_vpkm'] < 10) & (df['SPEED_kph'] < 20) |  # 低密度不应低速
            (df['DENSITY_vpkm'] > 80) & (df['SPEED_kph'] > 40) |  # 高密度不应高速
            (df['FLOW_vph'] > 2000) & (df['SPEED_kph'] < 10)      # 高流量不应极低速
        )
        
        df_cleaned = df[theory_mask].copy()
        removed_count = initial_count - len(df_cleaned)
        self.log_step("交通流理论约束", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _remove_insufficient_data_segments(self, df):
        """移除数据量不足的路段"""
        initial_count = len(df)
        
        # 计算每个路段的数据量
        segment_counts = df.groupby('ROADSECT_ID').size()
        sufficient_segments = segment_counts[segment_counts >= 100].index
        
        df_cleaned = df[df['ROADSECT_ID'].isin(sufficient_segments)].copy()
        removed_count = initial_count - len(df_cleaned)
        self.log_step("移除数据不足路段", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _smooth_time_series_outliers(self, df):
        """时间序列异常平滑处理（温和版本）"""
        initial_count = len(df)
        
        # 按路段和日期排序
        df_sorted = df.sort_values(['ROADSECT_ID', 'DATE', 'TIME']).copy()
        
        # 计算相邻时间点的变化率，使用更宽松的阈值
        for col in ['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']:
            # 按路段分组计算变化率
            df_sorted[f'{col}_change_rate'] = df_sorted.groupby('ROADSECT_ID')[col].pct_change().abs()
        
        # 使用更宽松的变化率阈值，只移除极端突变
        high_change_mask = (
            (df_sorted['SPEED_kph_change_rate'] > 1.5) |  # 从0.8改为1.5
            (df_sorted['FLOW_vph_change_rate'] > 2.0) |   # 从1.0改为2.0
            (df_sorted['DENSITY_vpkm_change_rate'] > 2.0)  # 从1.0改为2.0
        )
        
        # 移除突变点
        df_cleaned = df_sorted[~high_change_mask.fillna(False)].copy()
        
        # 清理临时列
        change_cols = [col for col in df_cleaned.columns if col.endswith('_change_rate')]
        df_cleaned = df_cleaned.drop(columns=change_cols)
        
        removed_count = initial_count - len(df_cleaned)
        self.log_step("时间序列突变处理(温和)", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _percentile_truncation(self, df):
        """分位数截断（温和版本）"""
        initial_count = len(df)
        
        # 使用0.5%-99.5%分位数截断，比1%-99%更宽松
        percentile_mask = True
        for col in ['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']:
            p05 = df[col].quantile(0.005)  # 0.5%分位数
            p995 = df[col].quantile(0.995)  # 99.5%分位数
            percentile_mask = percentile_mask & (df[col] >= p05) & (df[col] <= p995)
        
        df_cleaned = df[percentile_mask].copy()
        removed_count = initial_count - len(df_cleaned)
        self.log_step("分位数截断(0.5%-99.5%)", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _zscore_cleaning(self, df):
        """Z-score清洗"""
        initial_count = len(df)
        
        # 按路段计算Z-score
        zscore_mask = True
        
        for col in ['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']:
            # 计算每个路段的Z-score
            df[f'{col}_zscore'] = df.groupby('ROADSECT_ID')[col].transform(
                lambda x: np.abs(stats.zscore(x, nan_policy='omit'))
            )
            # 保留Z-score < 3.5的数据
            zscore_mask = zscore_mask & (df[f'{col}_zscore'] < 3.5)
        
        df_cleaned = df[zscore_mask].copy()
        
        # 清理临时列
        zscore_cols = [col for col in df_cleaned.columns if col.endswith('_zscore')]
        df_cleaned = df_cleaned.drop(columns=zscore_cols)
        
        removed_count = initial_count - len(df_cleaned)
        self.log_step("Z-score清洗", removed_count, len(df_cleaned))
        
        return df_cleaned
    
    def _print_cleaning_summary(self, original_df, cleaned_df):
        """打印清洗总结"""
        print("📋 数据清洗总结报告")
        print("=" * 60)
        
        total_removed = self.original_count - self.cleaned_count
        removal_rate = total_removed / self.original_count * 100
        
        print(f"原始数据量: {self.original_count:,} 条")
        print(f"清洗后数据量: {self.cleaned_count:,} 条")
        print(f"总移除数据: {total_removed:,} 条 ({removal_rate:.2f}%)")
        print(f"数据保留率: {100-removal_rate:.2f}%")
        print()
        
        print("📊 清洗前后数据质量对比:")
        print("-" * 40)
        
        for col in ['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']:
            print(f"\n{col}:")
            print(f"  清洗前: 均值={original_df[col].mean():.2f}, 标准差={original_df[col].std():.2f}")
            print(f"  清洗后: 均值={cleaned_df[col].mean():.2f}, 标准差={cleaned_df[col].std():.2f}")
            
            # 计算偏度和峰度改善
            original_skew = original_df[col].skew()
            cleaned_skew = cleaned_df[col].skew()
            original_kurt = original_df[col].kurtosis()
            cleaned_kurt = cleaned_df[col].kurtosis()
            
            print(f"  偏度改善: {original_skew:.3f} → {cleaned_skew:.3f}")
            print(f"  峰度改善: {original_kurt:.3f} → {cleaned_kurt:.3f}")
        
        print("\n🎯 异常值检测结果:")
        print("-" * 40)
        for col in ['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm']:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
            outlier_rate = outliers / len(cleaned_df) * 100
            print(f"  {col} 异常值比例: {outlier_rate:.2f}% ({outliers:,} 条)")

def main(gentle_mode=True):
    """
    主函数示例
    
    参数:
    gentle_mode (bool): True为温和模式（推荐），False为标准模式
    """
    # 创建清洗器实例
    cleaner = TrafficDataCleaner()
    
    # 读取数据（请根据实际文件路径修改）
    print("📖 正在读取数据...")
    df = pd.read_csv(r'E:\github_projects\math_modeling_playground\playground\路段速度数据_03.csv')  # 请替换为实际文件路径
    
    # 确保数据类型正确
    df['SPEED_kph'] = pd.to_numeric(df['SPEED_kph'], errors='coerce')
    df['FLOW_vph'] = pd.to_numeric(df['FLOW_vph'], errors='coerce')
    df['DENSITY_vpkm'] = pd.to_numeric(df['DENSITY_vpkm'], errors='coerce')
    
    # 移除缺失值
    df = df.dropna(subset=['SPEED_kph', 'FLOW_vph', 'DENSITY_vpkm'])
    
    # 执行清洗
    cleaned_df = cleaner.clean_traffic_data(df, gentle_mode=gentle_mode)
    
    # 保存清洗后的数据
    mode_suffix = "_gentle" if gentle_mode else "_standard"
    output_file = f'cleaned_traffic_data{mode_suffix}.csv'
    cleaned_df.to_csv(output_file, index=False)
    print(f"\n💾 清洗后的数据已保存至: {output_file}")
    
    return cleaned_df

if __name__ == "__main__":
    # 运行示例 - 推荐使用温和模式
    cleaned_data = main(gentle_mode=True)
    
    # 如果你已经有DataFrame，可以直接使用：
    # cleaner = TrafficDataCleaner()
    # cleaned_df = cleaner.clean_traffic_data(your_dataframe, gentle_mode=True)
    
    print("💡 使用建议:")
    print("- gentle_mode=True: 温和清洗，保留更多数据（推荐）")
    print("- gentle_mode=False: 标准清洗，数据质量更高但可能损失较多数据")
    pass