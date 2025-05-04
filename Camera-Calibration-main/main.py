import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import rcParams

# 设置matplotlib支持中文
rcParams['font.family'] = 'SimHei'  # 使用支持中文的字体SimHei
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 图片在当前文件夹的位置
file_in = 'p1'  # 原始图片存放位置
file_out = 'p3'  # 最后图片的保存位置

# 创建输出目录（如果不存在）
if not os.path.exists(file_out):
    os.makedirs(file_out)

# 棋盘格模板规格，只算内角点个数，不算最外面的一圈点
w = 9
h = 6

# 找棋盘格角点
# 世界坐标系中的棋盘格点，在张正友标定法中认为Z = 0
# mgrid创建了大小为9×6×2的三维矩阵，在reshape成二维以后赋给objp，objp最后为(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
objp = np.zeros((w * h, 3), np.float32)  # 大小为wh×3的0矩阵
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)  # :2是因为认为Z=0
objpoints = []  # 储存在世界坐标系中的三维点
imgpoints = []  # 储存在图像平面的二维点

images = os.listdir(file_in)  # 读入图像序列
i = 0
img_h = 0
img_w = 0

# 算法迭代的终止条件，第一项表示迭代次数达到最大次数时停止迭代，第二项表示角点位置变化的最小值已经达到最小时停止迭代
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
for fname in images:
    img = cv2.imread(file_in + '/' + fname)
    if img is None:
        print(f"无法读取图像 {fname}，跳过")
        continue

    img_h = np.size(img, 0)
    img_w = np.size(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # RGB转灰度
    # 找到棋盘格角点，存放角点于corners，如果找到足够点对，将其存储起来，ret为非零值
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # 检测到角点后，进行亚像素级别角点检测，更新角点
    if ret == True:
        i += 1
        # 输入图像gray；角点初始坐标corners；搜索窗口为2*winsize+1；表示窗口的最小（-1.-1）表示忽略；求角点的迭代终止条件
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)  # 空间坐标
        imgpoints.append(corners)  # 角点坐标即图像坐标
        # 角点显示
        corners_img = img.copy()
        cv2.drawChessboardCorners(corners_img, (w, h), corners, ret)
        # cv2.imshow('findCorners', img)
        cv2.imwrite(file_out + '/print_corners' + str(i) + '.jpg', corners_img)
        # cv2.waitKey(10)
# cv2.destroyAllWindows()

"""
求解参数
输入：世界坐标系里的位置；像素坐标；图像的像素尺寸大小；
输出：
ret: 重投影误差；
mtx：内参矩阵；
dist：畸变系数；
rvecs：旋转向量 （外参数）；
tvecs：平移向量 （外参数）；
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(("ret（重投影误差）:"), ret)
print(("mtx（内参矩阵）:\n"), mtx)
print(("dist（畸变参数）:\n"), dist)  # 5个畸变参数，(k_1,k_2,p_1,p_2,k_3)
print(("rvecs（旋转向量）:\n"), rvecs)
print(("tvecs（平移向量）:\n"), tvecs)

# 优化内参数和畸变系数
# 使用相机内参mtx和畸变系数dist，并使用cv.getOptimalNewCameraMatrix()
# 通过设定自由自由比例因子alpha。
# 当alpha设为0的时候，将会返回一个剪裁过的将去畸变后不想要的像素去掉的内参数和畸变系数；
# 当alpha设为1的时候，将会返回一系个包含额外黑色像素点的内参数和畸变数，并返回一个ROI用于将其剪裁掉。
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_w, img_h), 0, (img_w, img_h))

# 矫正畸变
if len(images) > 0:
    img2 = cv2.imread(file_in + '/' + images[0])
    dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
    cv2.imwrite(file_out + '/calibresult.jpg', dst)
    print("newcameramtx（优化后相机内参）:\n", newcameramtx)

# 反投影误差total_error,越接近0，说明结果越理想。
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)  # 计算三维点到二维图像的投影
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)  # 反投影得到的点与图像上检测到的点的误差
    total_error += error
print(("total error: "), total_error / len(objpoints))  # 记平均


# ------------- 以下是新增的增强现实功能代码 -------------

def get_projection_matrix(camera_matrix, rvec, tvec):
    """
    计算完整的投影矩阵 P = K[R|t]

    参数:
    camera_matrix - 相机内参矩阵
    rvec - 旋转向量
    tvec - 平移向量

    返回:
    projection_matrix - 3x4投影矩阵
    """
    # 将旋转向量转换为旋转矩阵
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # 构建外参矩阵 [R|t]
    extrinsic = np.hstack((rotation_matrix, tvec.reshape(-1, 1)))

    # 计算投影矩阵
    projection_matrix = np.dot(camera_matrix, extrinsic)
    return projection_matrix, rotation_matrix, tvec


def create_cube_points(size, position):
    """
    创建1:1:1立方体的顶点

    参数:
    size - 立方体边长
    position - 立方体在棋盘格坐标系中的位置 [x,y,z]

    返回:
    cube_points - 8个顶点的3D坐标
    """
    cube = np.array([
        [0, 0, 0],  # 0: 左下前
        [size, 0, 0],  # 1: 右下前
        [size, size, 0],  # 2: 右上前
        [0, size, 0],  # 3: 左上前
        [0, 0, size],  # 4: 左下后
        [size, 0, size],  # 5: 右下后
        [size, size, size],  # 6: 右上后
        [0, size, size]  # 7: 左上后
    ], dtype=np.float32)

    # 添加位置偏移
    cube[:, 0] += position[0]
    cube[:, 1] += position[1]
    cube[:, 2] += position[2]

    return cube


def project_cube(cube_points, camera_matrix, rvec, tvec, dist_coeffs):
    """
    将立方体投影到图像平面

    参数:
    cube_points - 立方体的3D顶点
    camera_matrix - 相机内参
    rvec - 旋转向量
    tvec - 平移向量
    dist_coeffs - 畸变系数

    返回:
    cube_2d - 投影后的2D点坐标
    """
    cube_2d, _ = cv2.projectPoints(cube_points, rvec, tvec, camera_matrix, dist_coeffs)
    return cube_2d.reshape(-1, 2)


def draw_cube(img, cube_2d, color=(0, 0, 255), thickness=2):
    """
    在图像上绘制立方体

    参数:
    img - 输入图像
    cube_2d - 立方体顶点的2D坐标
    color - 立方体颜色，默认红色 (BGR)
    thickness - 线条粗细

    返回:
    result_img - 绘制了立方体的图像
    """
    result_img = img.copy()

    # 定义立方体的边
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)  # 连接边
    ]

    # 将点转换为整数
    cube_2d_int = cube_2d.astype(np.int32)

    # 绘制边
    for edge in edges:
        pt1 = tuple(cube_2d_int[edge[0]])
        pt2 = tuple(cube_2d_int[edge[1]])
        cv2.line(result_img, pt1, pt2, color, thickness)

    # 绘制面（半透明）
    faces = [
        [0, 1, 2, 3],  # 底面
        [4, 5, 6, 7],  # 顶面
        [0, 1, 5, 4],  # 前面
        [2, 3, 7, 6],  # 后面
        [0, 3, 7, 4],  # 左面
        [1, 2, 6, 5]  # 右面
    ]

    # 创建一个覆盖层绘制半透明面
    overlay = result_img.copy()
    for face in faces:
        pts = np.array([cube_2d_int[i] for i in face])
        cv2.fillPoly(overlay, [pts], color)

    # 混合图像以获得半透明效果
    alpha = 0.3  # 透明度
    result_img = cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0)

    return result_img


def visualize_3d_cube(cube_points, title="3D Cube"):
    """
    可视化3D立方体

    参数:
    cube_points - 立方体的3D顶点
    title - 图标题
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 定义立方体的边
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)  # 连接边
    ]

    # 绘制边
    for edge in edges:
        xs = [cube_points[edge[0], 0], cube_points[edge[1], 0]]
        ys = [cube_points[edge[0], 1], cube_points[edge[1], 1]]
        zs = [cube_points[edge[0], 2], cube_points[edge[1], 2]]
        ax.plot(xs, ys, zs, 'r-', linewidth=2)

    # 设置轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # 设置坐标轴范围以保持比例相同
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # 计算轴范围中点
    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    # 计算最大范围以确保缩放一致
    max_range = max([
        x_limits[1] - x_limits[0],
        y_limits[1] - y_limits[0],
        z_limits[1] - z_limits[0]
    ]) / 2.0

    # 设置新的轴范围
    ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
    ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
    ax.set_zlim3d([z_middle - max_range, z_middle + max_range])

    return fig


# 生成增强现实图像
print("\n创建增强现实图像...")

# 选择至少3个不同角度
# 如果检测到的图像少于3个，我们使用所有可用的图像
selected_indices = list(range(min(3, len(objpoints))))
cube_size = 1.0  # 1:1:1立方体边长

# 我们选择三个不同的立方体位置来展示
cube_positions = [
    [4, 3, 0],  # 中心位置（正好在棋盘格上）
    [2, 1, 0],  # 左下位置
    [6, 4, 0]  # 右上位置
]

# 为每个选择的视角生成增强现实图像
for i, idx in enumerate(selected_indices):
    # 确保索引有效
    if idx >= len(images):
        print(f"警告：索引 {idx} 超出范围，跳过")
        continue

    # 读取图像
    img_path = file_in + '/' + images[idx]
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图像 {img_path}，跳过")
        continue

    # 选择立方体位置（使用循环位置）
    position = cube_positions[i % len(cube_positions)]

    # 创建立方体
    cube_points = create_cube_points(cube_size, position)

    # 计算投影矩阵
    proj_matrix, rotation_matrix, t_vec = get_projection_matrix(mtx, rvecs[idx], tvecs[idx])
    print(f"\n视角 {i + 1} 投影矩阵:")
    print(proj_matrix)

    # 投影立方体
    cube_2d = project_cube(cube_points, mtx, rvecs[idx], tvecs[idx], dist)

    print(f"\n视角 {i + 1} 立方体3D位置:")
    print(cube_points)
    print(f"\n视角 {i + 1} 立方体2D投影位置:")
    print(cube_2d)

    # 绘制立方体
    ar_img = draw_cube(img, cube_2d)

    # 保存增强现实图像
    ar_img_path = file_out + f"/ar_image_{i + 1}.jpg"
    cv2.imwrite(ar_img_path, ar_img)
    print(f"增强现实图像已保存到 {ar_img_path}")

    # 创建结果可视化
    plt.figure(figsize=(15, 10))

    # 原始图像
    plt.subplot(221)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'视角 {i + 1} - 原始图像')
    plt.axis('off')

    # 增强现实图像
    plt.subplot(222)
    plt.imshow(cv2.cvtColor(ar_img, cv2.COLOR_BGR2RGB))
    plt.title(f'视角 {i + 1} - 增强现实图像')
    plt.axis('off')

    # 3D立方体可视化
    ax = plt.subplot(223, projection='3d')
    # 绘制立方体
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)  # 连接边
    ]
    for edge in edges:
        xs = [cube_points[edge[0], 0], cube_points[edge[1], 0]]
        ys = [cube_points[edge[0], 1], cube_points[edge[1], 1]]
        zs = [cube_points[edge[0], 2], cube_points[edge[1], 2]]
        ax.plot(xs, ys, zs, 'r-', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'视角 {i + 1} - 3D立方体')

    # 标定参数
    plt.subplot(224)
    plt.axis('off')
    param_text = (
        f"相机内参矩阵:\n{np.round(mtx, 2)}\n\n"
        f"立方体位置: {position}\n\n"
        f"投影矩阵:\n{np.round(proj_matrix, 2)}"
    )
    plt.text(0.5, 0.5, param_text, ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
             family='monospace', multialignment='left', transform=plt.gca().transAxes)

    # 保存可视化结果
    result_path = file_out + f"/result_visualization_{i + 1}.jpg"
    plt.tight_layout()
    plt.savefig(result_path)
    # plt.show()  # 如果要显示图像，请取消注释
    plt.close()
    print(f"结果可视化已保存到 {result_path}")

print("\n增强现实任务完成！")
print(f"所有结果文件已保存到 {file_out} 目录。")