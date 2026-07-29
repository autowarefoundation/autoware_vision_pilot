# 通过地面棋盘格进行相机外参标定与单应性映射
# Extrinsic Camera Calibration & Homography Mapping via Ground Checkerboards

脚本 **calc_front_camera_homography.py** 提供了一个高度鲁棒的自动化 Python 工具，用于计算**单应性矩阵（$H$）**，将 2D 相机图像坐标 $(u, v)$ 映射到扁平的 3D 真实世界道路坐标 $(X, Y)$。
The script **calc_front_camera_homography.py** provides a highly robust, automated Python tool to calculate the **Homography Matrix ($H$)** that maps 2D camera image coordinates $(u, v)$ to flat 3D real-world road coordinates $(X, Y)$.

通过精确使用**四个 2×2 地面棋盘格标记**，该脚本自动以亚像素精度检测标定目标交叉点，将它们相对于物理坐标进行结构化，求解单应性系统，并输出直接投影到相机帧上的可视化验证网格。
By leveraging exactly **four 2x2 ground-plane checkerboard markers**, the script automatically detects calibration target intersections with sub-pixel accuracy, structures them relative to physical coordinates, solves the homography system, and outputs a visual validation grid backprojected directly onto your camera frame.

## 1. 物理标定设置
## 1. Physical Calibration Setup

要建立从像素到物理道路坐标的高精度映射，你必须在车辆前方的沥青路面上放置**四个 2×2 黑白棋盘格目标**，使其在相机视野内 — 请参考下方的"地面布局图"。
To establish a highly precise mapping from pixels to physical road coordinates, you must place four **2x2 black-and-white checkerboard targets** flat on the asphalt surface in front of the vehicle, in view of the camera - please see the 'Ground Layout Diagram' below for reference.

### 地面布局图
### Ground Layout Diagram

![Calibration Setup Guide](camera_calibration_setup.jpg)

### 坐标映射参考
### Coordinate Mapping Reference

| 目标位置 / Target Location | 图像坐标标签 / Image Coord Label | 世界坐标 $(X, Y)$ / World Coordinates |
| :--- | :---: | :---: |
| **左上** / **Top-Left** | $(u_1, v_1)$ | $(X1, Y1)$ |
| **右上** / **Top-Right** | $(u_2, v_2)$ | $(X2, Y2)$ |
| **左下** / **Bottom-Left** | $(u_3, v_3)$ | $(X3, Y3)$ |
| **右下** / **Bottom-Right** | $(u_4, v_4)$ | $(X4, Y4)$ |

**请注意坐标轴约定 — X 轴正方向向前，Y 轴正方向向左，原点位于车辆前保险杠中心的地面上**
**Please note the axis convention - X is positive forward and Y is positive left with the origin at the front bumper at the centre of the vehicle on the ground**

### 地面标记放置步骤
### Step-by-Step Ground Marker Placement

1. **打印四个 2×2 棋盘格目标：**
   1. **Print Four 2x2 Checkerboard Targets:**
   - 2×2 棋盘格由两个黑色和两个白色方块在中间交汇组成。
   - A 2x2 checkerboard consists of two black and two white squares meeting in the middle.
   - **关键细节：** 中心的精确交叉点作为单个像素级坐标 $(u,v)$。
   - **Crucial Detail:** The exact intersection point at the center serves as the single pixel-accurate coordinate $(u,v)$.

2. **在地面上布置网格：**
   2. **Lay Out the Grid on the Ground:**
   - 使用 A4 纸，打印四份[棋盘格图案](checkerboard-bw.png)，每页应有一个 2×2 棋盘格图案。将棋盘格以矩形模式放置在车辆前方的平坦沥青上，使相机可见，并用胶带将页面牢固地固定在路面上。
   - Using an A4 paper, print out four copies of the [checkerboard grid pattern](checkerboard-bw.png), each page should have one 2x2 checkerboard pattern on it. Place the checkerboards on the flat asphalt in front of the vehicle in a rectangular pattern such that they are visible to the camera and firmly tape the pages flat to the road surface.
   - 确保它们不会滑动或变形。
   - Ensure they do not slide or warp.

3. **测量你的坐标：**
   3. **Measure Your Coordinates:**
   - 选择车辆前保险杠中心作为**世界坐标原点 $(0,0)$**。
   - Pick the centre of the front bumper of your vehicle to act as your **World Coordinate Origin $(0,0)$**
   - 测量从世界坐标原点到四个棋盘格中心的精确物理距离并记录下来，确保记下哪个测量值属于哪个棋盘格（左上、右上、左下、右下），如"地面布局图"所示。
   - Measure the precise physical distance from the World Coordinate Origin to the four checkerboard centres and record them, ensuring you write down which of the checkerboards the measurement belongs to (top-left, top-right, bottom-left, and bottom-right) as referenced in the 'Ground Layout Diagram'.

4. **保存标定图像：**
   4. **Save a calibration image:**
   - 保存一张相机能看到沥青上棋盘格图案的场景图像，该图像将作为脚本的输入标定图像。
   - Save an image of the scene in which the camera sees the checkerboard patterns on the asphalt, this image will be used by the script as the input calibration image.
   - 确保相机牢固且刚性地安装 — 如果在运行标定脚本后相机发生移动，标定参数将失效，你需要重新运行标定脚本。
   - Ensure that camera is firmly and rigidly mounted - if the camera moves after you have run the calibration script, the calibration parameters will be invalid and you will need to re-run the calibration script again.

---

## 2. 脚本工作原理
## 2. How the Script Works

该流水线通过五个不同阶段执行：
The pipeline executes through five distinct stages:

1. **亚像素角点提取：** 执行 OpenCV 的 `cv2.findChessboardCorners`，目标模式大小为 `(1,1)`（检测单个 2×2 棋盘格交叉点）。然后使用 `cv2.cornerSubPix` 将坐标细化到亚像素精度。
1. **Sub-Pixel Corner Extraction:** OpenCV's `cv2.findChessboardCorners` is executed with a target pattern size of `(1,1)` (which detects a single 2x2 checkerboard intersection). The coordinate is then refined down to sub-pixel accuracy using `cv2.cornerSubPix`.

2. **迭代检测与空间掩码：** 由于标准图像有多个相同目标，脚本进行迭代搜索。一旦定位到目标中心，它就用半径 $R = \max(\text{width}, \text{height})/20$ 的实心白圆将该区域从灰度搜索空间中掩码掉，以确保后续迭代锁定不同的棋盘格。
2. **Iterative Detection & Space Masking:** Because a standard image has multiple identical targets, the script searches iteratively. Once it localizes a target center, it masks that region out of the grayscale search space with a solid white circle of radius $R = \max(\text{width}, \text{height})/20$ to ensure subsequent iterations lock onto different boards.

3. **鲁棒空间排序：** 四个检测到的点根据其图像坐标自动聚类并映射：
3. **Robust Spatial Sorting:** The four detected points are automatically clustered and mapped using their image coordinates:
   - **行（上 vs 下）：** 坐标按其 $v$ 像素位置垂直排序。最接近地平线（较小 $v$ 值）的两个点形成"上"行；最接近相机（较大 $v$ 值）的两个点形成"下"行。
   - **Rows (Top vs. Bottom):** Coordinates are sorted vertically by their $v$ pixel locations. The two points closest to the horizon line (smaller $v$ values) form the "top" row; the two closest to the camera (larger $v$ values) form the "bottom" row.
   - **列（左 vs 右）：** 在每行内，点按其 $u$ 像素位置水平排序（较小 $u$ 为左，较大 $u$ 为右）。
   - **Columns (Left vs. Right):** Inside each row, the points are sorted horizontally by their $u$ pixel locations (lower $u$ is Left, higher $u$ is Right).

4. **单应性求解器：** 使用排序后的图像坐标 $(u, v)$ 和 CLI 提供的对应物理坐标 $(X,Y)$，OpenCV 使用直接线性变换（DLT）算法计算 $3 \times 3$ 单应性矩阵 $H$：
4. **Homography Solver:** Using the sorted image coordinates $(u, v)$ and your corresponding CLI-provided physical coordinates $(X,Y)$, OpenCV computes the $3 \times 3$ homography matrix $H$ using the Direct Linear Transform (DLT) algorithm:
   $$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
   该矩阵以序列化的 `FileStorage` 配置标准（`H.yaml`）原生写入。
   This matrix is written natively into a serialized `FileStorage` configuration standard (`H.yaml`).

5. **反向反投影叠加：** 脚本取逆单应性 $H^{-1}$，将世界坐标边界 $[0 \le X \le W, 0 \le Y \le L]$ 的数学完美均匀网格投影回透视相机图像空间。它将这些渲染为绿色虚拟车道和水平环，以展示标定对齐效果。
5. **Inverse Backprojection Overlay:** The script takes the inverse homography $H^{-1}$ to project a mathematically perfect uniform grid from world coordinate boundaries $[0 \le X \le W, 0 \le Y \le L]$ back onto the perspective camera image space. It renders these as green virtual lanes and horizontal rings to demonstrate the calibration alignment.

---

## 3. 运行脚本
## 3. Run the script

通过输入捕获的相机帧以及从真实世界标记网格测量的物理坐标来执行脚本，如下所示 — 确保将 H 矩阵保存到 [VisionPilot/config](../VisionPilot/config/) 文件夹中，并**替换默认的 H.yaml 文件**：
Execute the script by feeding it your captured camera frame along with the physical coordinates measured from your real-world marker grid, an example is shown below - ensure to save your H-matrix in the [VisionPilot/config](../VisionPilot/config/) folder and **replace the default H.yaml file** with your new yaml file.

```bash
python calc_homography_2x2.py --img road_frame.jpg \
  --out ../VisionPilot/config/H.yaml \
  --tl 0.0 15.0 \
  --tr 3.7 15.0 \
  --bl 0.0 0.0 \
  --br 3.7 0.0
```

### 命令行参数
### Commandline Arguments

- **--img** — 捕获的源标定图像文件路径 / Path to the captured source calibration image file.
- **--out** — 评估后的单应性矩阵应保存的目标文件路径 / Target file path where the evaluated Homography matrix should be saved.
- **--tl** — 左上标记世界坐标：X（深度）、Y（水平偏移）/ Top-Left Marker World Coordinates: X (depth), Y (horizontal offset).
- **--tr** — 右上标记世界坐标：X（深度）、Y（水平偏移）/ Top-Right Marker World Coordinates: X (depth), Y (horizontal offset).
- **--bl** — 左下标记世界坐标：X（深度）、Y（水平偏移）/ Bottom-Left Marker World Coordinates: X (depth), Y (horizontal offset).
- **--br** — 右下标记世界坐标：X（深度）、Y（水平偏移）/ Bottom-Right Marker World Coordinates: X (depth), Y (horizontal offset).

---

### 文件输出（H.yaml）
### File Output (H.yaml)

单应性矩阵已保存，可供 Vision Pilot 使用。
The Homography matrix is saved and can be used by Vision Pilot

### 可视化输出
### Visualization Output

脚本自动生成叠加可视化图像，保存为 `<your_out_name>_visualization.png`。
The script automatically produces an overlay visualization image saved as <your_out_name>_visualization.png.

**绿线：** 表示绘制在路面上的均匀物理网格，投影回透视中。如果你的标定准确，这些线将与现有道路线完美平行，并在接近地平线时正确压缩。
**Green Lines:** Represent a uniform physical grid drawn on the road floor, projected back into perspective. If your calibration is accurate, these lines will align perfectly parallel to existing road lines, and compress correctly as they approach the horizon.

**红圈：** 指示已识别的棋盘格中心，带有文本标签（左上、右上等），确认正确的识别配对。
**Red Circles:** Indicate the identified centers of your checkerboards, printed with text labels (Top-Left, Top-Right, etc.) confirming the correct identification pairing

---

### 故障排除与最佳实践
### Troubleshooting & Best Practices

#### 未找到角点
#### No Corners Found

确保道路上有高对比度照明。直接投射在棋盘格上的阴影可能导致角点检测失败。
Ensure high-contrast illumination on the road. Shadows cast directly across a checkerboard can cause corner detection to fail.

如果你有反光路面，调整 `cv2.findChessboardCorners` 标志。
Adjust cv2.findChessboardCorners flags if you have reflective pavement.

#### 极端相机倾斜
#### Extreme Camera Tilts

空间排序假设相机的侧倾角最小。如果相机侧倾超过 45 度，垂直分离逻辑可能混淆左右配对。在标定期间保持相机水平。
Spatial sorting assumes the camera has minimal roll. If the camera is tilted sideways by more than 45 degrees, the vertical separation logic may mix up the left-right pairing. Keep the camera level during calibration.

#### 线条扭曲到天空（地平线错误）
#### Lines Distorting Into Sky (Horizon Errors)

投影超过地平线的直线可能在数学上"环绕"并错误渲染。代码使用自定义透视深度过滤器（`homog_img[:, 2] > 1e-5`）安全地裁剪延伸到无限空间的线段。
Straight lines projected past the horizon line can mathematically "wrap around" and render incorrectly. The code utilizes a custom perspective-depth filter (homog_img[:, 2] > 1e-5) to clip segments extending into infinite space safely.
