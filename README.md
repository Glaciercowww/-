# 基于棋盘格的增强现实系统教程

本教程介绍如何完成一个基于单目视觉的增强现实系统，在棋盘格图像上放置虚拟立方体。完整代码整合已有的相机标定部分，并增加了所需的3D立方体投影和多角度渲染功能。。

## 项目概述

项目分为三个主要部分：
1. 相机标定 
2. 三维立方体投影 
3. 多角度增强现实图像生成 

## 数据文件结构

```
项目目录/
├── p1/           # 输入图像目录（原始棋盘格图像）
└── p3/           # 输出结果目录（自动创建）
    ├── print_corners*.jpg              # 角点检测结果
    ├── calibresult.jpg                 # 校正后的图像
    ├── ar_image_*.jpg                  # 增强现实图像
    └── result_visualization_*.jpg      # 结果可视化
```

## 实现细节

### 1. 相机标定 

在这部分中，我们使用棋盘格图像来计算相机内参和畸变系数：

```python
# 找棋盘格角点
ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
if ret == True:
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    objpoints.append(objp)
    imgpoints.append(corners)
```

标定相机并获取参数：

```python
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
```

计算完整的投影矩阵：

```python
def get_projection_matrix(camera_matrix, rvec, tvec):
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    
    # 构建外参矩阵 [R|t]
    extrinsic = np.hstack((rotation_matrix, tvec.reshape(-1, 1)))
    
    # 计算投影矩阵
    projection_matrix = np.dot(camera_matrix, extrinsic)
    return projection_matrix, rotation_matrix, t_vec
```

### 2. 三维立方体投影 

创建1:1:1比例的立方体并设置位置：

```python
def create_cube_points(size, position):
    cube = np.array([
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
        [0, 0, size], [size, 0, size], [size, size, size], [0, size, size]
    ], dtype=np.float32)
    
    # 添加位置偏移
    cube[:, 0] += position[0]
    cube[:, 1] += position[1]
    cube[:, 2] += position[2]
    
    return cube
```

使用投影变换将3D点投影到2D图像平面：

```python
def project_cube(cube_points, camera_matrix, rvec, tvec, dist_coeffs):
    cube_2d, _ = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist_coeffs)
    return cube_2d.reshape(-1, 2)
```

### 3. 多角度增强现实图像 

在图像上绘制立方体：

```python
def draw_cube(img, cube_2d, color=(0, 0, 255), thickness=2):
    result_img = img.copy()
    
    # 定义立方体的边
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)   # 连接边
    ]
    
    # 绘制边和半透明面
    # ...
```

生成多个不同角度的增强现实图像：

```python
# 选择至少3个不同角度
selected_indices = list(range(min(3, len(objpoints))))

# 三个不同的立方体位置
cube_positions = [
    [4, 3, 0],  # 中心位置
    [2, 1, 0],  # 左下位置
    [6, 4, 0]   # 右上位置
]

# 为每个角度生成增强现实图像
for i, idx in enumerate(selected_indices):
    # ...
```

## 运行指南

1. 准备数据：
   - 将至少3张不同角度拍摄的棋盘格图像放入`p1`目录

2. 运行代码：
   ```bash
   python ar_system.py
   ```

3. 检查结果：
   - 相机标定参数（内参矩阵、畸变系数）
   - 投影矩阵
   - 立方体3D位置和2D投影位置
   - 增强现实图像（至少3个不同角度）

## 数学原理

### 相机模型

相机投影矩阵 P 将3D点转换为2D点：
```
P = K[R|t]
```
其中：
- K：相机内参矩阵（3×3）
- R：旋转矩阵（3×3）
- t：平移向量（3×1）

### 投影变换

3D点到2D图像平面的投影：
```
s·[u,v,1]^T = K·[R|t]·[X,Y,Z,1]^T
```
其中：
- (u,v)：投影后的像素坐标
- (X,Y,Z)：世界坐标
- s：比例因子

## 关键参数解释

1. 内参矩阵（Camera Matrix）：
   ```
   K = [[fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]]
   ```
   - fx, fy: 焦距（以像素为单位）
   - cx, cy: 光心坐标（主点）

2. 畸变系数（Distortion Coefficients）：
   ```
   dist = [k1, k2, p1, p2, k3]
   ```
   - k1, k2, k3: 径向畸变系数
   - p1, p2: 切向畸变系数

3. 投影矩阵（Projection Matrix）：
   ```
   P = K·[R|t]  # 3×4矩阵
   ```

## 扩展功能

完成基本要求后，可以考虑以下扩展：

1. 实时视频流中添加虚拟立方体
2. 添加更复杂的3D模型（而不仅仅是立方体）
3. 添加光照、阴影和材质
4. 实现基于标记的物体跟踪
5. 实现无标记增强现实

## 常见问题排除

1. 角点检测失败：
   - 确保棋盘格图像清晰可见
   - 尝试调整亮度和对比度
   - 确保棋盘格完全在图像中

2. 投影不准确：
   - 检查相机标定质量（反投影误差）
   - 确保立方体位置正确设置
   - 检查坐标系一致性

3. 渲染问题：
   - 检查2D点是否在图像范围内
   - 确保绘制顺序正确（先绘制远处的面）

## 结论

通过这个项目，您已经实现了一个完整的增强现实系统，从相机标定到3D物体投影和多角度渲染。这些技术是更复杂AR应用的基础，如交互式AR、物体识别和跟踪等。
