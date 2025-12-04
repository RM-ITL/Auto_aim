import re
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

log_file = 'logs/2025-12-03_20-07-29.log'

x_coords = []
y_coords = []

pattern = r'重投影中心点: \(([0-9.]+), ([0-9.]+)\)'

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            x_coords.append(x)
            y_coords.append(y)

print(f"Total data points: {len(x_coords)}")
print(f"X range: [{min(x_coords):.2f}, {max(x_coords):.2f}]")
print(f"Y range: [{min(y_coords):.2f}, {max(y_coords):.2f}]")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.scatter(x_coords, y_coords, c=range(len(x_coords)), cmap='viridis', s=2, alpha=0.6)
ax1.set_xlabel('X Coordinate (pixels)', fontsize=12)
ax1.set_ylabel('Y Coordinate (pixels)', fontsize=12)
ax1.set_title('Reprojection Center Distribution', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.invert_yaxis()

ax2.plot(x_coords, y_coords, linewidth=0.5, alpha=0.7)
ax2.scatter(x_coords[0], y_coords[0], c='green', s=100, marker='o', label='Start', zorder=5)
ax2.scatter(x_coords[-1], y_coords[-1], c='red', s=100, marker='x', label='End', zorder=5)
ax2.set_xlabel('X Coordinate (pixels)', fontsize=12)
ax2.set_ylabel('Y Coordinate (pixels)', fontsize=12)
ax2.set_title('Reprojection Center Trajectory', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('reprojection_visualization.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved: reprojection_visualization.png")

fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8))

ax3.plot(range(len(x_coords)), x_coords, linewidth=0.8, color='blue', alpha=0.7)
ax3.set_xlabel('Time Series', fontsize=12)
ax3.set_ylabel('X Coordinate (pixels)', fontsize=12)
ax3.set_title('X Coordinate over Time', fontsize=14)
ax3.grid(True, alpha=0.3)

ax4.plot(range(len(y_coords)), y_coords, linewidth=0.8, color='red', alpha=0.7)
ax4.set_xlabel('Time Series', fontsize=12)
ax4.set_ylabel('Y Coordinate (pixels)', fontsize=12)
ax4.set_title('Y Coordinate over Time', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reprojection_time_series.png', dpi=300, bbox_inches='tight')
print("Time series plot saved: reprojection_time_series.png")

plt.show()
