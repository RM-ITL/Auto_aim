import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ArmorStabilityAnalyzer:
    def __init__(self, csv_file):
        """
        初始化分析器
        Args:
            csv_file: CSV文件路径
        """
        self.df = pd.read_csv(csv_file)
        self.armor_corners = []
        self.light_corners = []
        
    def extract_coordinates(self):
        """
        从CSV中提取装甲板和灯条的角点坐标
        """
        # 提取装甲板四个角点坐标
        armor_x_cols = [col for col in self.df.columns if 'armor' in col and '/x' in col]
        armor_y_cols = [col for col in self.df.columns if 'armor' in col and '/y' in col]
        
        # 提取灯条四个角点坐标
        light_x_cols = [col for col in self.df.columns if 'light_corners' in col and '/x' in col]
        light_y_cols = [col for col in self.df.columns if 'light_corners' in col and '/y' in col]
        
        # 组织数据为 [(x0,y0), (x1,y1), (x2,y2), (x3,y3)] 的格式
        for i in range(len(self.df)):
            armor_points = []
            light_points = []
            
            # 提取装甲板角点
            for j in range(min(len(armor_x_cols), len(armor_y_cols))):
                x = self.df.iloc[i][armor_x_cols[j]]
                y = self.df.iloc[i][armor_y_cols[j]]
                armor_points.append((x, y))
            
            # 提取灯条角点
            for j in range(min(len(light_x_cols), len(light_y_cols))):
                x = self.df.iloc[i][light_x_cols[j]]
                y = self.df.iloc[i][light_y_cols[j]]
                light_points.append((x, y))
            
            self.armor_corners.append(armor_points)
            self.light_corners.append(light_points)
    
    def calculate_jitter_metrics(self, corners_list):
        """
        计算抖动指标
        Args:
            corners_list: 角点列表 [[(x0,y0), ...], ...]
        Returns:
            dict: 包含各种抖动指标的字典
        """
        metrics = {}
        
        # 转换为numpy数组便于计算
        corners_array = np.array(corners_list)
        n_frames = len(corners_list)
        n_corners = len(corners_list[0])
        
        # 计算每个角点的统计指标
        for corner_idx in range(n_corners):
            x_coords = corners_array[:, corner_idx, 0]
            y_coords = corners_array[:, corner_idx, 1]
            
            # 标准差（抖动程度）
            metrics[f'corner_{corner_idx}_std_x'] = np.std(x_coords)
            metrics[f'corner_{corner_idx}_std_y'] = np.std(y_coords)
            
            # 变化范围
            metrics[f'corner_{corner_idx}_range_x'] = np.max(x_coords) - np.min(x_coords)
            metrics[f'corner_{corner_idx}_range_y'] = np.max(y_coords) - np.min(y_coords)
            
            # 平均绝对偏差
            metrics[f'corner_{corner_idx}_mad_x'] = np.mean(np.abs(x_coords - np.mean(x_coords)))
            metrics[f'corner_{corner_idx}_mad_y'] = np.mean(np.abs(y_coords - np.mean(y_coords)))
        
        # 计算中心点的稳定性
        centers = np.mean(corners_array, axis=1)  # 每帧的中心点
        metrics['center_std_x'] = np.std(centers[:, 0])
        metrics['center_std_y'] = np.std(centers[:, 1])
        
        # 计算帧间变化
        if n_frames > 1:
            frame_diffs = []
            for i in range(1, n_frames):
                diff = np.sqrt(np.sum((corners_array[i] - corners_array[i-1])**2))
                frame_diffs.append(diff)
            metrics['mean_frame_diff'] = np.mean(frame_diffs)
            metrics['max_frame_diff'] = np.max(frame_diffs)
        
        return metrics
    
    def plot_trajectory(self):
        """
        绘制角点轨迹图
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Corner Point Trajectories', fontsize=14)
        
        # 装甲板角点轨迹
        ax1 = axes[0, 0]
        for i in range(4):
            x_coords = [corners[i][0] for corners in self.armor_corners if i < len(corners)]
            y_coords = [corners[i][1] for corners in self.armor_corners if i < len(corners)]
            ax1.plot(x_coords, y_coords, 'o-', markersize=3, 
                    label=f'Corner {i}', alpha=0.7)
        ax1.set_title('Armor Corners Trajectory')
        ax1.set_xlabel('X Coordinate (pixels)')
        ax1.set_ylabel('Y Coordinate (pixels)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 灯条角点轨迹
        ax2 = axes[0, 1]
        for i in range(4):
            x_coords = [corners[i][0] for corners in self.light_corners if i < len(corners)]
            y_coords = [corners[i][1] for corners in self.light_corners if i < len(corners)]
            ax2.plot(x_coords, y_coords, 'o-', markersize=3, 
                    label=f'Corner {i}', alpha=0.7)
        ax2.set_title('Light Corners Trajectory')
        ax2.set_xlabel('X Coordinate (pixels)')
        ax2.set_ylabel('Y Coordinate (pixels)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 装甲板中心点轨迹
        ax3 = axes[1, 0]
        armor_centers = [np.mean(corners, axis=0) for corners in self.armor_corners if len(corners) > 0]
        if armor_centers:
            armor_centers = np.array(armor_centers)
            ax3.plot(armor_centers[:, 0], armor_centers[:, 1], 'ro-', 
                    markersize=4, linewidth=1, alpha=0.7)
            ax3.set_title('Armor Center Trajectory')
            ax3.set_xlabel('X Coordinate (pixels)')
            ax3.set_ylabel('Y Coordinate (pixels)')
            ax3.grid(True, alpha=0.3)
        
        # 灯条中心点轨迹
        ax4 = axes[1, 1]
        light_centers = [np.mean(corners, axis=0) for corners in self.light_corners if len(corners) > 0]
        if light_centers:
            light_centers = np.array(light_centers)
            ax4.plot(light_centers[:, 0], light_centers[:, 1], 'bo-', 
                    markersize=4, linewidth=1, alpha=0.7)
            ax4.set_title('Light Center Trajectory')
            ax4.set_xlabel('X Coordinate (pixels)')
            ax4.set_ylabel('Y Coordinate (pixels)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_jitter_distribution(self):
        """
        绘制抖动分布图
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Jitter Distribution Analysis', fontsize=14)
        
        # 计算帧间位移
        armor_displacements = []
        light_displacements = []
        
        for i in range(1, len(self.armor_corners)):
            if len(self.armor_corners[i]) > 0 and len(self.armor_corners[i-1]) > 0:
                for j in range(min(len(self.armor_corners[i]), len(self.armor_corners[i-1]))):
                    dx = self.armor_corners[i][j][0] - self.armor_corners[i-1][j][0]
                    dy = self.armor_corners[i][j][1] - self.armor_corners[i-1][j][1]
                    armor_displacements.append(np.sqrt(dx**2 + dy**2))
        
        for i in range(1, len(self.light_corners)):
            if len(self.light_corners[i]) > 0 and len(self.light_corners[i-1]) > 0:
                for j in range(min(len(self.light_corners[i]), len(self.light_corners[i-1]))):
                    dx = self.light_corners[i][j][0] - self.light_corners[i-1][j][0]
                    dy = self.light_corners[i][j][1] - self.light_corners[i-1][j][1]
                    light_displacements.append(np.sqrt(dx**2 + dy**2))
        
        # 装甲板位移分布
        ax1 = axes[0, 0]
        if armor_displacements:
            ax1.hist(armor_displacements, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax1.set_title('Armor Frame-to-Frame Displacement')
            ax1.set_xlabel('Displacement (pixels)')
            ax1.set_ylabel('Frequency')
            ax1.grid(True, alpha=0.3)
        
        # 灯条位移分布
        ax2 = axes[0, 1]
        if light_displacements:
            ax2.hist(light_displacements, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('Light Frame-to-Frame Displacement')
            ax2.set_xlabel('Displacement (pixels)')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
        
        # 装甲板坐标标准差
        ax3 = axes[1, 0]
        armor_metrics = self.calculate_jitter_metrics(self.armor_corners)
        std_values = [armor_metrics[key] for key in armor_metrics if 'std' in key and 'center' not in key]
        if std_values:
            ax3.bar(range(len(std_values)), std_values, alpha=0.7, color='red')
            ax3.set_title('Armor Coordinate Standard Deviations')
            ax3.set_xlabel('Corner & Axis Index')
            ax3.set_ylabel('Standard Deviation (pixels)')
            ax3.grid(True, alpha=0.3)
        
        # 灯条坐标标准差
        ax4 = axes[1, 1]
        light_metrics = self.calculate_jitter_metrics(self.light_corners)
        std_values = [light_metrics[key] for key in light_metrics if 'std' in key and 'center' not in key]
        if std_values:
            ax4.bar(range(len(std_values)), std_values, alpha=0.7, color='orange')
            ax4.set_title('Light Coordinate Standard Deviations')
            ax4.set_xlabel('Corner & Axis Index')
            ax4.set_ylabel('Standard Deviation (pixels)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_report(self):
        """
        生成稳定性分析报告
        """
        print("="*60)
        print("Armor Detection Stability Analysis Report")
        print("="*60)
        
        # 装甲板分析
        print("\n--- Armor Corners Analysis ---")
        armor_metrics = self.calculate_jitter_metrics(self.armor_corners)
        
        print(f"Center Point Jitter:")
        print(f"  X-axis std: {armor_metrics.get('center_std_x', 0):.3f} pixels")
        print(f"  Y-axis std: {armor_metrics.get('center_std_y', 0):.3f} pixels")
        
        print(f"\nFrame-to-Frame Changes:")
        print(f"  Mean displacement: {armor_metrics.get('mean_frame_diff', 0):.3f} pixels")
        print(f"  Max displacement: {armor_metrics.get('max_frame_diff', 0):.3f} pixels")
        
        # 灯条分析
        print("\n--- Light Corners Analysis ---")
        light_metrics = self.calculate_jitter_metrics(self.light_corners)
        
        print(f"Center Point Jitter:")
        print(f"  X-axis std: {light_metrics.get('center_std_x', 0):.3f} pixels")
        print(f"  Y-axis std: {light_metrics.get('center_std_y', 0):.3f} pixels")
        
        print(f"\nFrame-to-Frame Changes:")
        print(f"  Mean displacement: {light_metrics.get('mean_frame_diff', 0):.3f} pixels")
        print(f"  Max displacement: {light_metrics.get('max_frame_diff', 0):.3f} pixels")
        
        print("\n" + "="*60)
    
    def run_analysis(self, save_plots=True):
        """
        运行完整的稳定性分析
        """
        print("Loading and processing data...")
        self.extract_coordinates()
        
        print("Generating analysis report...")
        self.generate_report()
        
        print("\nCreating visualization plots...")
        fig1 = self.plot_trajectory()
        fig2 = self.plot_jitter_distribution()
        
        if save_plots:
            fig1.savefig('armor_trajectory_analysis.png', dpi=300, bbox_inches='tight')
            fig2.savefig('armor_jitter_distribution.png', dpi=300, bbox_inches='tight')
            print("Plots saved successfully!")
        
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 请将这里替换为您的CSV文件路径
    csv_file_path = "/home/guo/ITL_sentry_auto/corners.csv"
    
    # 创建分析器实例
    analyzer = ArmorStabilityAnalyzer(csv_file_path)
    
    # 运行分析
    analyzer.run_analysis(save_plots=True)
