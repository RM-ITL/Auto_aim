import numpy as np
import plotly.graph_objects as go

# ============================================================================
# 核心变换矩阵：从相机坐标系到world坐标系的旋转矩阵
# ============================================================================
# 这个矩阵描述了world坐标系的三个轴在相机坐标系中的方向
# 矩阵的每一列代表world坐标系的一个轴（X、Y、Z）在相机坐标系中的方向向量
R_camera_to_world = np.array([
    [0.000, 0.000, 1.000],   # 第1列：world-X = Camera-Z (相机光轴方向)
    [-1.000, 0.000, 0.000],   # 第2列：world-Y = -Camera-X (相机右侧的反向)
    [0.000, -1.000, 0.000]     # 第3列：world-Z = -Camera-Y (相机下方的反向)
])


def create_axis_arrows(R, origin, scale=0.5, name_prefix='', colors=None):
    """
    创建三维坐标轴的可视化箭头
    
    这个函数会为每个坐标轴创建一个带箭头的线段，帮助我们直观地看到坐标系的方向
    
    参数说明：
        R: 旋转矩阵，每一列代表该坐标系的一个轴的方向向量
        origin: 坐标系的原点位置（三维坐标）
        scale: 坐标轴的长度，用于控制箭头的显示大小
        name_prefix: 坐标系的名称前缀，如'Camera'或'IMU'
        colors: 三个坐标轴的颜色列表，分别对应X、Y、Z轴
    
    返回：
        包含所有图形元素的列表，可以直接添加到Plotly图形中
    """
    if colors is None:
        colors = ['red', 'green', 'blue']
    
    labels = ['X', 'Y', 'Z']
    arrows = []
    
    # 为每个坐标轴（X、Y、Z）创建可视化元素
    for i in range(3):
        # 从旋转矩阵中提取当前轴的方向向量，并按比例缩放
        direction = R[:, i] * scale
        end_point = origin + direction
        
        # 创建箭头的主体部分（一条线段）
        arrow = go.Scatter3d(
            x=[origin[0], end_point[0]],
            y=[origin[1], end_point[1]],
            z=[origin[2], end_point[2]],
            mode='lines+text',
            line=dict(color=colors[i], width=10),
            text=['', f'{name_prefix}-{labels[i]}'],  # 在箭头末端显示标签
            textposition='top center',
            textfont=dict(size=16, color=colors[i], family='Arial Black'),
            name=f'{name_prefix}-{labels[i]}',
            showlegend=True,
            # 鼠标悬停时显示详细的方向向量信息
            hovertemplate=f'{name_prefix}-{labels[i]}<br>方向: [{R[0,i]:.3f}, {R[1,i]:.3f}, {R[2,i]:.3f}]<extra></extra>'
        )
        arrows.append(arrow)
        
        # 在箭头末端添加一个锥体，使箭头方向更加明显
        cone = go.Cone(
            x=[end_point[0]],
            y=[end_point[1]],
            z=[end_point[2]],
            u=[direction[0] * 0.25],  # 锥体指向的方向
            v=[direction[1] * 0.25],
            w=[direction[2] * 0.25],
            colorscale=[[0, colors[i]], [1, colors[i]]],  # 使用与箭头相同的颜色
            showscale=False,
            showlegend=False,
            sizemode='absolute',
            sizeref=0.2
        )
        arrows.append(cone)
    
    return arrows


# ============================================================================
# 创建主图形对象
# ============================================================================
fig = go.Figure()

# 定义两个坐标系的原点位置（都在同一点，便于比较）
origin = np.array([0, 0, 0])

# 相机坐标系：使用标准的单位矩阵，表示相机自身的坐标系
# 在相机坐标系中：X轴向右，Y轴向下，Z轴向前（光轴方向）
R_camera = np.eye(3)
camera_colors = ['rgb(255, 100, 0)', 'rgb(0, 180, 255)', 'rgb(255, 0, 150)']

# 将相机坐标轴添加到图形中
camera_axes = create_axis_arrows(R_camera, origin, scale=0.5, 
                                 name_prefix='相机', colors=camera_colors)
for axis in camera_axes:
    fig.add_trace(axis)

# world坐标系：使用我们定义的旋转矩阵来表示world坐标系
# 这个旋转矩阵告诉我们world的每个轴在相机坐标系中指向哪里
world_colors = ['rgb(255, 200, 0)', 'rgb(50, 255, 50)', 'rgb(150, 50, 255)']

world_axes = create_axis_arrows(R_camera_to_world, origin, scale=0.5, 
                              name_prefix='IMU', colors=world_colors)
for axis in world_axes:
    fig.add_trace(axis)

# 在原点位置添加一个标记点，表示两个坐标系的公共原点
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers+text',
    marker=dict(size=12, color='black', symbol='diamond'),
    text=['原点'],
    textposition='bottom center',
    textfont=dict(size=14, family='Arial Black'),
    name='坐标系原点',
    showlegend=True
))

# ============================================================================
# 设置图形的布局和视角
# ============================================================================
fig.update_layout(
    title=dict(
        text='相机坐标系 ↔ world坐标系 变换关系<br><sub>两个坐标系共享同一个原点，展示它们之间的旋转关系</sub>',
        x=0.5,
        xanchor='center',
        font=dict(size=20, family='Arial')
    ),
    scene=dict(
        # 设置三个坐标轴的显示范围和样式
        xaxis=dict(
            title='相机 X轴方向 (右)',
            range=[-0.8, 0.8],
            backgroundcolor="rgb(250, 250, 250)",
            gridcolor="rgb(220, 220, 220)"
        ),
        yaxis=dict(
            title='相机 Y轴方向 (下)',
            range=[-0.8, 0.8],
            backgroundcolor="rgb(250, 250, 250)",
            gridcolor="rgb(220, 220, 220)"
        ),
        zaxis=dict(
            title='相机 Z轴方向 (前/光轴)',
            range=[-0.8, 0.8],
            backgroundcolor="rgb(250, 250, 250)",
            gridcolor="rgb(220, 220, 220)"
        ),
        aspectmode='cube',  # 保持立方体比例，避免变形
        # 设置观察视角，从一个能看清所有轴的角度观察
        camera=dict(
            eye=dict(x=1.3, y=1.3, z=1.3),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)  # 设置Z轴为向上方向
        )
    ),
    showlegend=True,
    legend=dict(
        x=0.02, 
        y=0.98,
        font=dict(size=11, family='Arial'),
        bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='black',
        borderwidth=2
    ),
    width=1200,
    height=900
)

# ============================================================================
# 添加变换矩阵的数学表示
# ============================================================================
matrix_text = (
    f'<b>旋转矩阵 R (相机→world):</b><br><br>'
    f'[{R_camera_to_world[0,0]:6.3f}  {R_camera_to_world[0,1]:6.3f}  {R_camera_to_world[0,2]:6.3f}]<br>'
    f'[{R_camera_to_world[1,0]:6.3f}  {R_camera_to_world[1,1]:6.3f}  {R_camera_to_world[1,2]:6.3f}]<br>'
    f'[{R_camera_to_world[2,0]:6.3f}  {R_camera_to_world[2,1]:6.3f}  {R_camera_to_world[2,2]:6.3f}]<br><br>'
    '<i>矩阵的每一列表示world坐标系<br>的一个轴在相机坐标系中的方向</i>'
)

fig.add_annotation(
    text=matrix_text,
    xref="paper", yref="paper",
    x=0.98, y=0.98,
    showarrow=False,
    align='left',
    bgcolor='rgba(255, 250, 200, 0.95)',
    bordercolor='darkorange',
    borderwidth=2,
    font=dict(family='Courier New', size=11)
)

# ============================================================================
# 添加坐标系对应关系的详细说明
# ============================================================================
relationship_text = (
    '<b>坐标轴对应关系：</b><br><br>'
    '<b>IMU-X轴</b> = 相机-Z轴<br>'
    '  (朝向相机的光轴方向)<br>'
    f'  方向向量: [{R_camera_to_world[0,0]:.1f}, {R_camera_to_world[1,0]:.1f}, {R_camera_to_world[2,0]:.1f}]<br><br>'
    '<b>IMU-Y轴</b> = -相机-X轴<br>'
    '  (与相机右侧方向相反)<br>'
    f'  方向向量: [{R_camera_to_world[0,1]:.1f}, {R_camera_to_world[1,1]:.1f}, {R_camera_to_world[2,1]:.1f}]<br><br>'
    '<b>IMU-Z轴</b> = -相机-Y轴<br>'
    '  (与相机向下方向相反)<br>'
    f'  方向向量: [{R_camera_to_world[0,2]:.1f}, {R_camera_to_world[1,2]:.1f}, {R_camera_to_world[2,2]:.1f}]'
)

fig.add_annotation(
    text=relationship_text,
    xref="paper", yref="paper",
    x=0.02, y=0.65,
    showarrow=False,
    align='left',
    bgcolor='rgba(230, 240, 255, 0.95)',
    bordercolor='royalblue',
    borderwidth=2,
    font=dict(family='Arial', size=11)
)

# ============================================================================
# 添加颜色编码说明
# ============================================================================
color_legend = (
    '<b>颜色图例：</b><br><br>'
    '<span style="color:rgb(255,100,0)">●</span> 相机-X (右侧)<br>'
    '<span style="color:rgb(0,180,255)">●</span> 相机-Y (下方)<br>'
    '<span style="color:rgb(255,0,150)">●</span> 相机-Z (前方/光轴)<br><br>'
    '<span style="color:rgb(255,200,0)">●</span> IMU-X<br>'
    '<span style="color:rgb(50,255,50)">●</span> IMU-Y<br>'
    '<span style="color:rgb(150,50,255)">●</span> IMU-Z'
)

fig.add_annotation(
    text=color_legend,
    xref="paper", yref="paper",
    x=0.02, y=0.35,
    showarrow=False,
    align='left',
    bgcolor='rgba(255, 255, 255, 0.95)',
    bordercolor='gray',
    borderwidth=2,
    font=dict(family='Arial', size=11)
)

# 添加使用说明
usage_info = (
    '<b>交互提示：</b><br><br>'
    '• 鼠标拖动可旋转视角<br>'
    '• 滚轮缩放<br>'
    '• 双击重置视角<br>'
    '• 鼠标悬停查看详细信息'
)

fig.add_annotation(
    text=usage_info,
    xref="paper", yref="paper",
    x=0.02, y=0.12,
    showarrow=False,
    align='left',
    bgcolor='rgba(240, 255, 240, 0.95)',
    bordercolor='green',
    borderwidth=2,
    font=dict(family='Arial', size=10)
)

# 显示图形
fig.show()