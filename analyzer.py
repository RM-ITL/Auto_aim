import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 如果系统有中文字体，可以改为 'SimHei' 或 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False

class PredictionAccuracyAnalyzer:
    """预测准确率分析器 - 只分析armor_id为4的数据"""
    
    def __init__(self, pnp_file: str, predict_file: str, time_offset: float = 0.05):
        """
        初始化分析器
        
        参数:
            pnp_file: 解算端数据文件路径
            predict_file: 预测端数据文件路径
            time_offset: 预测时间偏移量（默认0.5秒）
        """
        self.time_offset = time_offset
        self.tolerance = 0.05  # 时间匹配容差（50ms）
        self.target_armor_id = 4  # 只分析armor_id为4的数据
        
        # 读取数据
        print("正在读取数据文件...")
        self.pnp_data = pd.read_csv(pnp_file)
        self.predict_data = pd.read_csv(predict_file)
        
        print(f"原始解算端数据: {len(self.pnp_data)} 条记录")
        print(f"原始预测端数据: {len(self.predict_data)} 条记录")
        
        # 首先检查数据结构
        self._inspect_data_structure()
        
        # 过滤只保留armor_id为4的数据
        self._filter_armor_id()
        
    def _inspect_data_structure(self):
        """检查数据结构，找出实际的armor_id列名"""
        print("\n检查数据结构...")
        
        # 查找包含armor_id的列
        pnp_armor_cols = [col for col in self.pnp_data.columns if 'armor_id' in col.lower()]
        predict_armor_cols = [col for col in self.predict_data.columns if 'armor_id' in col.lower()]
        
        print(f"解算端数据中的armor_id相关列: {pnp_armor_cols}")
        print(f"预测端数据中的armor_id相关列: {predict_armor_cols}")
        
        # 如果找到了armor_id列，打印一些样本值
        if pnp_armor_cols:
            print(f"解算端armor_id样本值: {self.pnp_data[pnp_armor_cols[0]].value_counts().head()}")
        if predict_armor_cols:
            print(f"预测端armor_id样本值: {self.predict_data[predict_armor_cols[0]].value_counts().head()}")
        
    def _filter_armor_id(self):
        """过滤数据，只保留armor_id为4的记录"""
        print(f"\n正在过滤数据，只保留armor_id={self.target_armor_id}的记录...")
        
        # 查找实际的armor_id列名
        pnp_armor_cols = [col for col in self.pnp_data.columns if 'armor_id' in col.lower()]
        predict_armor_cols = [col for col in self.predict_data.columns if 'armor_id' in col.lower()]
        
        # 过滤解算端数据
        if pnp_armor_cols:
            # 使用找到的第一个armor_id列
            armor_col = pnp_armor_cols[0]
            print(f"使用解算端armor_id列: {armor_col}")
            self.pnp_data = self.pnp_data[self.pnp_data[armor_col] == self.target_armor_id].copy()
        else:
            print("警告：解算端数据中未找到任何armor_id列，跳过过滤")
        
        # 过滤预测端数据
        if predict_armor_cols:
            # 使用找到的第一个armor_id列
            armor_col = predict_armor_cols[0]
            print(f"使用预测端armor_id列: {armor_col}")
            self.predict_data = self.predict_data[self.predict_data[armor_col] == self.target_armor_id].copy()
        else:
            print("警告：预测端数据中未找到任何armor_id列，跳过过滤")
        
        # 重置索引，确保索引连续
        self.pnp_data = self.pnp_data.reset_index(drop=True)
        self.predict_data = self.predict_data.reset_index(drop=True)
        
        print(f"过滤后解算端数据: {len(self.pnp_data)} 条记录")
        print(f"过滤后预测端数据: {len(self.predict_data)} 条记录")
        
        if len(self.pnp_data) == 0 or len(self.predict_data) == 0:
            print(f"警告：过滤后数据为空！请检查数据中是否存在armor_id={self.target_armor_id}的记录")
    
    def find_matching_pairs(self) -> List[Tuple[int, int]]:
        """
        查找匹配的数据对
        
        返回:
            匹配的索引对列表 [(pnp_index, predict_index), ...]
        """
        print(f"\n正在查找匹配的数据对（预测时间偏移: {self.time_offset}秒）...")
        print(f"只匹配armor_id={self.target_armor_id}的数据")
        
        matching_pairs = []
        
        # 确保时间列存在
        if '__time' not in self.pnp_data.columns or '__time' not in self.predict_data.columns:
            print("错误：数据中缺少__time列！")
            return matching_pairs
        
        # 对预测数据按时间排序
        predict_times = self.predict_data['__time'].values
        
        for pnp_idx, pnp_row in self.pnp_data.iterrows():
            pnp_time = pnp_row['__time']
            target_time = pnp_time + self.time_offset
            
            # 在预测数据中查找最接近目标时间的记录
            time_diffs = np.abs(predict_times - target_time)
            min_diff_idx = np.argmin(time_diffs)
            min_diff = time_diffs[min_diff_idx]
            
            # 如果时间差在容差范围内，认为是匹配的
            if min_diff <= self.tolerance:
                matching_pairs.append((pnp_idx, min_diff_idx))
        
        print(f"找到 {len(matching_pairs)} 对匹配的数据（armor_id={self.target_armor_id}）")
        
        # 验证匹配对的有效性
        max_pnp_idx = len(self.pnp_data) - 1
        max_predict_idx = len(self.predict_data) - 1
        
        valid_pairs = []
        for pnp_idx, predict_idx in matching_pairs:
            if 0 <= pnp_idx <= max_pnp_idx and 0 <= predict_idx <= max_predict_idx:
                valid_pairs.append((pnp_idx, predict_idx))
            else:
                print(f"警告：无效的索引对 ({pnp_idx}, {predict_idx})")
        
        if len(valid_pairs) < len(matching_pairs):
            print(f"过滤掉 {len(matching_pairs) - len(valid_pairs)} 个无效的匹配对")
        
        return valid_pairs
    
    def calculate_position_error(self, pnp_idx: int, predict_idx: int) -> Dict[str, float]:
        """
        计算位置预测误差
        
        参数:
            pnp_idx: 解算数据索引
            predict_idx: 预测数据索引
            
        返回:
            包含各项误差的字典
        """
        try:
            pnp_row = self.pnp_data.iloc[pnp_idx]
            predict_row = self.predict_data.iloc[predict_idx]
        except IndexError as e:
            print(f"索引错误: pnp_idx={pnp_idx}, predict_idx={predict_idx}")
            print(f"pnp_data长度: {len(self.pnp_data)}, predict_data长度: {len(self.predict_data)}")
            raise e
        
        # 提取解算端的位置数据
        pnp_x = pnp_row.get('/pnp_solver/targets/targets[0]/position/x', np.nan)
        pnp_y = pnp_row.get('/pnp_solver/targets/targets[0]/position/y', np.nan)
        pnp_z = pnp_row.get('/pnp_solver/targets/targets[0]/position/z', np.nan)
        
        # 提取预测端的位置数据
        pred_x = predict_row.get('/predictor/armor_predictions/predictions[0]/position/x', np.nan)
        pred_y = predict_row.get('/predictor/armor_predictions/predictions[0]/position/y', np.nan)
        pred_z = predict_row.get('/predictor/armor_predictions/predictions[0]/position/z', np.nan)
        
        # 计算误差
        error_x = abs(pred_x - pnp_x) if pd.notna(pred_x) and pd.notna(pnp_x) else np.nan
        error_y = abs(pred_y - pnp_y) if pd.notna(pred_y) and pd.notna(pnp_y) else np.nan
        error_z = abs(pred_z - pnp_z) if pd.notna(pred_z) and pd.notna(pnp_z) else np.nan
        
        # 计算欧氏距离误差
        if pd.notna(error_x) and pd.notna(error_y) and pd.notna(error_z):
            euclidean_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        else:
            euclidean_error = np.nan
        
        return {
            'error_x': error_x,
            'error_y': error_y,
            'error_z': error_z,
            'euclidean_error': euclidean_error,
            'pnp_distance': pnp_row.get('/pnp_solver/targets/targets[0]/distance', np.nan),
            'pred_distance': predict_row.get('/predictor/armor_predictions/predictions[0]/distance', np.nan)
        }
    
    def calculate_angle_error(self, pnp_idx: int, predict_idx: int) -> Dict[str, float]:
        """
        计算角度预测误差
        
        参数:
            pnp_idx: 解算数据索引
            predict_idx: 预测数据索引
            
        返回:
            包含角度误差的字典
        """
        try:
            pnp_row = self.pnp_data.iloc[pnp_idx]
            predict_row = self.predict_data.iloc[predict_idx]
        except IndexError as e:
            print(f"索引错误: pnp_idx={pnp_idx}, predict_idx={predict_idx}")
            raise e
        
        # 提取解算端的角度数据
        pnp_yaw = pnp_row.get('/pnp_solver/targets/targets[0]/yaw', np.nan)
        pnp_pitch = pnp_row.get('/pnp_solver/targets/targets[0]/pitch', np.nan)
        
        # 提取预测端的角度数据
        pred_yaw = predict_row.get('/predictor/armor_predictions/predictions[0]/yaw', np.nan)
        pred_pitch = predict_row.get('/predictor/armor_predictions/predictions[0]/pitch', np.nan)
        
        # 计算角度误差（注意处理角度环绕）
        def angle_diff(a1, a2):
            """计算两个角度之间的最小差值"""
            if pd.isna(a1) or pd.isna(a2):
                return np.nan
            diff = abs(a1 - a2)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            return diff
        
        yaw_error = angle_diff(pred_yaw, pnp_yaw)
        pitch_error = angle_diff(pred_pitch, pnp_pitch)
        
        return {
            'yaw_error': yaw_error,
            'pitch_error': pitch_error
        }
    
    def analyze_accuracy(self) -> Dict[str, any]:
        """
        分析预测准确率
        
        返回:
            包含分析结果的字典
        """
        if len(self.pnp_data) == 0 or len(self.predict_data) == 0:
            print(f"错误：没有armor_id={self.target_armor_id}的数据可供分析！")
            return None
            
        matching_pairs = self.find_matching_pairs()
        
        if not matching_pairs:
            print("没有找到匹配的数据对！")
            return None
        
        # 收集所有误差数据
        position_errors = []
        angle_errors = []
        
        print(f"\n正在计算armor_id={self.target_armor_id}的预测误差...")
        
        # 添加进度显示
        total_pairs = len(matching_pairs)
        for i, (pnp_idx, predict_idx) in enumerate(matching_pairs):
            if i % 100 == 0:
                print(f"  处理进度: {i}/{total_pairs} ({i/total_pairs*100:.1f}%)")
            
            try:
                pos_error = self.calculate_position_error(pnp_idx, predict_idx)
                angle_error = self.calculate_angle_error(pnp_idx, predict_idx)
                
                position_errors.append(pos_error)
                angle_errors.append(angle_error)
            except Exception as e:
                print(f"处理第{i}对数据时出错: {e}")
                continue
        
        # 转换为DataFrame以便分析
        pos_df = pd.DataFrame(position_errors)
        angle_df = pd.DataFrame(angle_errors)
        
        # 过滤掉NaN值
        pos_df = pos_df.dropna()
        angle_df = angle_df.dropna()
        
        print(f"有效位置数据: {len(pos_df)} 条")
        print(f"有效角度数据: {len(angle_df)} 条")
        
        if len(pos_df) == 0 or len(angle_df) == 0:
            print("错误：没有有效的数据可供分析！")
            return None
        
        # 计算统计指标
        results = {
            'armor_id': self.target_armor_id,
            'matching_pairs': len(matching_pairs),
            'valid_position_pairs': len(pos_df),
            'valid_angle_pairs': len(angle_df),
            
            # 位置误差统计
            'position_error_stats': {
                'x': {
                    'mean': pos_df['error_x'].mean(),
                    'std': pos_df['error_x'].std(),
                    'max': pos_df['error_x'].max(),
                    'percentile_95': pos_df['error_x'].quantile(0.95)
                },
                'y': {
                    'mean': pos_df['error_y'].mean(),
                    'std': pos_df['error_y'].std(),
                    'max': pos_df['error_y'].max(),
                    'percentile_95': pos_df['error_y'].quantile(0.95)
                },
                'z': {
                    'mean': pos_df['error_z'].mean(),
                    'std': pos_df['error_z'].std(),
                    'max': pos_df['error_z'].max(),
                    'percentile_95': pos_df['error_z'].quantile(0.95)
                },
                'euclidean': {
                    'mean': pos_df['euclidean_error'].mean(),
                    'std': pos_df['euclidean_error'].std(),
                    'max': pos_df['euclidean_error'].max(),
                    'percentile_95': pos_df['euclidean_error'].quantile(0.95)
                }
            },
            
            # 角度误差统计（转换为度）
            'angle_error_stats': {
                'yaw': {
                    'mean_deg': np.degrees(angle_df['yaw_error'].mean()),
                    'std_deg': np.degrees(angle_df['yaw_error'].std()),
                    'max_deg': np.degrees(angle_df['yaw_error'].max()),
                    'percentile_95_deg': np.degrees(angle_df['yaw_error'].quantile(0.95))
                },
                'pitch': {
                    'mean_deg': np.degrees(angle_df['pitch_error'].mean()),
                    'std_deg': np.degrees(angle_df['pitch_error'].std()),
                    'max_deg': np.degrees(angle_df['pitch_error'].max()),
                    'percentile_95_deg': np.degrees(angle_df['pitch_error'].quantile(0.95))
                }
            },
            
            # 原始数据
            'position_errors_df': pos_df,
            'angle_errors_df': angle_df
        }
        
        return results
    
    def print_results(self, results: Dict[str, any]):
        """打印分析结果"""
        print("\n" + "="*60)
        print(f"预测准确率分析结果 (armor_id = {self.target_armor_id})")
        print("="*60)
        
        print(f"\n匹配统计:")
        print(f"  - 找到匹配对数: {results['matching_pairs']}")
        print(f"  - 有效位置数据对: {results['valid_position_pairs']}")
        print(f"  - 有效角度数据对: {results['valid_angle_pairs']}")
        
        print(f"\n位置预测误差统计:")
        for axis in ['x', 'y', 'z', 'euclidean']:
            stats = results['position_error_stats'][axis]
            print(f"\n  {axis.upper()}轴误差:" if axis != 'euclidean' else "\n  欧氏距离误差:")
            print(f"    - 平均值: {stats['mean']:.4f} m")
            print(f"    - 标准差: {stats['std']:.4f} m")
            print(f"    - 最大值: {stats['max']:.4f} m")
            print(f"    - 95%分位数: {stats['percentile_95']:.4f} m")
        
        print(f"\n角度预测误差统计:")
        for angle in ['yaw', 'pitch']:
            stats = results['angle_error_stats'][angle]
            print(f"\n  {angle.upper()}角误差:")
            print(f"    - 平均值: {stats['mean_deg']:.2f}°")
            print(f"    - 标准差: {stats['std_deg']:.2f}°")
            print(f"    - 最大值: {stats['max_deg']:.2f}°")
            print(f"    - 95%分位数: {stats['percentile_95_deg']:.2f}°")
    
    def plot_results(self, results: Dict[str, any]):
        """绘制分析结果图表"""
        pos_df = results['position_errors_df']
        angle_df = results['angle_errors_df']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Prediction Accuracy Analysis (armor_id = {self.target_armor_id})', fontsize=16)
        
        # 位置误差直方图
        ax1 = axes[0, 0]
        ax1.hist(pos_df['euclidean_error'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Euclidean Error (m)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Position Prediction Error Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 各轴误差箱线图
        ax2 = axes[0, 1]
        pos_df[['error_x', 'error_y', 'error_z']].boxplot(ax=ax2)
        ax2.set_ylabel('Error (m)')
        ax2.set_title('Position Error by Axis')
        ax2.grid(True, alpha=0.3)
        
        # 角度误差直方图
        ax3 = axes[1, 0]
        ax3.hist(np.degrees(angle_df['yaw_error']), bins=30, alpha=0.5, label='Yaw', color='red')
        ax3.hist(np.degrees(angle_df['pitch_error']), bins=30, alpha=0.5, label='Pitch', color='green')
        ax3.set_xlabel('Angle Error (degrees)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Angle Prediction Error Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 误差随距离变化
        ax4 = axes[1, 1]
        ax4.scatter(pos_df['pnp_distance'], pos_df['euclidean_error'], alpha=0.5, s=10)
        ax4.set_xlabel('Distance (m)')
        ax4.set_ylabel('Euclidean Error (m)')
        ax4.set_title('Prediction Error vs Distance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'prediction_accuracy_analysis_armor{self.target_armor_id}.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    # 创建分析器实例（只分析armor_id=4的数据）
    analyzer = PredictionAccuracyAnalyzer('pnp.csv', 'predict.csv', time_offset=0.035)
    
    # 执行分析
    results = analyzer.analyze_accuracy()
    
    if results:
        # 打印结果
        analyzer.print_results(results)
        
        # 绘制图表
        analyzer.plot_results(results)
        
        # 保存详细结果到CSV
        results['position_errors_df'].to_csv(f'position_errors_analysis_armor{analyzer.target_armor_id}.csv', index=False)
        results['angle_errors_df'].to_csv(f'angle_errors_analysis_armor{analyzer.target_armor_id}.csv', index=False)
        
        print(f"\n分析完成！结果已保存到:")
        print(f"  - prediction_accuracy_analysis_armor{analyzer.target_armor_id}.png (图表)")
        print(f"  - position_errors_analysis_armor{analyzer.target_armor_id}.csv (位置误差详细数据)")
        print(f"  - angle_errors_analysis_armor{analyzer.target_armor_id}.csv (角度误差详细数据)")
    else:
        print("\n分析失败！请检查数据文件和armor_id设置。")


if __name__ == "__main__":
    main()
