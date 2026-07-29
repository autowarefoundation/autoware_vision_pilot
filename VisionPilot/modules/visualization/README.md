# 可视化模块
# VISUALIZATION MODULE

## 致谢
## Acknowledgement

感谢 [Ethan](https://dev.to/ethand91) 及其博客文章 [使用 C++ WebRTC GStreamer 流式传输相机](https://dev.to/ethand91/streaming-camera-with-c-webrtc-gstreamer-pof)。
I would like to thank [Ethan](https://dev.to/ethand91) and his blog post of [Streaming Camera with C++ WebRTC GStreamer](https://dev.to/ethand91/streaming-camera-with-c-webrtc-gstreamer-pof).

你的实现对我完成本模块确实非常有帮助和启发。
Your implementation was truly helpful and inspiring for me to complete this module.

## I. 概述
## I. Overview

WebRTC 可视化模块通过 WebRTC 协议为 VisionPilot 流水线提供实时视频流功能。它提供以下核心功能：
The WebRTC Visualization Module provides a real-time video streaming capability for the VisionPilot pipeline via WebRTC protocol. It serves the following core functions:

1. **实时帧捕获和编码**：接受 OpenCV `cv::Mat` 帧并通过 GStreamer 将其编码为 VP8 视频编解码器。
1. **Real-time frame capture and encoding** which accepts OpenCV `cv::Mat` frames and encodes them to VP8 video codec via GStreamer.
2. **WebRTC 点对点流式传输**：在服务器（VisionPilot 应用）和浏览器客户端之间建立 WebRTC 对等连接，实现通过互联网或局域网的实时视频传输。
2. **WebRTC peer-to-peer streaming** which establishes a WebRTC peer connection between the server (VisionPilot app) and browser clients, enabling live video delivery over the internet or LAN.
3. **轻量级浏览器客户端**：提供一个最小化的自包含 HTML5 页面，内置 WebRTC JavaScript 客户端，浏览器无需外部依赖。
3. **Lightweight browser client** which serves a minimal, self-contained HTML5 page with built-in WebRTC JavaScript client without external dependencies required for the browser.
4. 实现基于 WebSocket 的信令，用于 SDP（会话描述协议）提供/应答协商和 ICE（交互式连接建立）候选交换。
4. Implements WebSocket-based signaling for SDP (Session Description Protocol) offer/answer negotiation and ICE (Interactive Connectivity Establishment) candidate exchange.
5. **线程安全的帧流式传输**：管理来自主应用线程的并发帧推送，同时在单独的线程中运行 GStreamer 流水线和事件循环。
5. **Thread-safe frame streaming** that manages concurrent frame pushes from the main application thread while running a GStreamer pipeline and event loop in separate threads.

本模块对于开发和测试阶段的自主驾驶流水线的远程监控、调试和可视化至关重要。
This module is essential for downstream remote monitoring, debugging, and visualization of autonomous driving pipelines during development and testing phases.

## II. 架构与模块结构
## II. Architecture && Module structure

### 1. 架构
### 1. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VisionPilot 应用                              │
│                    VisionPilot Application                      │
│                  (vision_pilot.cpp 主线程)                       │
│                  (vision_pilot.cpp main thread)                 │
│                                                                 │
│   数据采集                                                       │
│   DATA CAPTURE                                                  │
│   ┌──────────────────────┐                                      │
│   │  V4L2/ROS2 相机       │                                      │
│   │      源               │                                      │
│   │      Source          │                                      │
│   └──────────┬───────────┘                                      │
│              │                                                  │
│              │ cv::Mat 帧 (33ms 循环)                            │
│              │ cv::Mat frames (33ms loop)                       │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────────────────────────┐                  │
│   │  (各种上游模块，如                        │                  │
│   │  模型推理、处理、计算等)                   │                  │
│   │  (Various upstream modules, like         │                  │
│   │  model inference, processing, calc etc.) │                  │
│   └──────────┬───────────────────────────────┘                  │
│              │                                                  │
│              │ 帧 + 纵向/横向规划结果                            │
│              │ Frames & longitudinal/lateral planning results   │
│              │                                                  │
│              ▼                                                  │
│   可视化                                                         │
│   VISUALIZATION                                                 │
│   ┌──────────────────────────────────────────┐                  │
│   │  visualization::render_frame()           │                  │
│   │  (绘制帧 + 规划结果)                      │                  │
│   │  (draw frame + planning results)         │                  │
│   └──────────┬───────────────────────────────┘                  │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────────────────────────┐                  │
│   │  WebRTCStreamer::push_frame()            │                  │
│   │  (通过 WebRTC 流式传输到端点)             │                  │
│   │  (stream to endpoint via WebRTC)         │                  │
│   └──────────┬───────────────────────────────┘                  │
│              │                                                  │
└──────────────┼──────────────────────────────────────────────────┘
               │
               │ BGR 帧 + 元数据
               │ BGR frames + metadata
               │
    ┌──────────▼───────────────────────────────────────────┐
    │         WebRTCStreamer::Impl（内部实现）              │
    │         WebRTCStreamer::Impl (Internal)              │
    │                                                      │
    │  ┌────────────────────────────────────────────────┐  │
    │  │  GStreamer 流水线（单独线程）                    │  │
    │  │  GStreamer Pipeline (separate thread)          │  │
    │  │                                                │  │
    │  │  appsrc => queue => videoconvert => vp8enc =>  │  │
    │  │  rtpvp8pay => webrtcbin                        │  │
    │  │                                                │  │
    │  │  ┌──────────────────────────────────────────┐  │  │
    │  │  │ WebRTC 对等连接（GStreamer）              │  │  │
    │  │  │ WebRTC peer connection (GStreamer)       │  │  │
    │  │  │  - 管理媒体流                            │  │  │
    │  │  │  - 管理媒体流 / Manages media stream     │  │  │
    │  │  │  - 生成 SDP 提供                         │  │  │
    │  │  │  - 生成 SDP 提供 / Generates SDP offers  │  │  │
    │  │  │  - 收集 ICE 候选                         │  │  │
    │  │  │  - 收集 ICE 候选 / Gathers ICE candidates│  │  │
    │  │  └──────────────────────────────────────────┘  │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                      │
    │  ┌────────────────────────────────────────────────┐  │
    │  │  信令层（SoupServer + WebSocket）               │  │
    │  │  Signaling Layer (SoupServer + WebSocket)      │  │
    │  │                                                │  │
    │  │  HTTP 处理器：                                  │  │
    │  │  HTTP handler:                                 │  │
    │  │    GET / => 提供 kBrowserHtml                  │  │
    │  │    GET / => serves kBrowserHtml                │  │
    │  │                                                │  │
    │  │  WebSocket 处理器：                             │  │
    │  │  WebSocket handler:                            │  │
    │  │    - 接收：SDP 应答、ICE 候选                   │  │
    │  │    - 接收：SDP 应答、ICE 候选 / Receives: SDP answer, ICE candidates │  │
    │  │    - 发送：SDP 提供、ICE 候选                   │  │
    │  │    - 发送：SDP 提供、ICE 候选 / Sends: SDP offer, ICE candidates   │  │
    │  │    - 队列 + 刷新机制用于排序                    │  │
    │  │    - 队列 + 刷新机制用于排序 / Queue + flush mechanism for ordering│  │
    │  │                                                │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                      │
    └───────────────┬──────────────────────────────────────┘
                    │
        ┌───────────┴──────────────┐
        │                          │
        ▼                          ▼
   ┌─────────────────┐        ┌──────────────────┐
   │  浏览器客户端    │        │  网络             │
   │  Browser Client │        │  Network         │
   │  (HTML5 + JS)   │ <====> │  (Internet/LAN)  │
   │                 │        └──────────────────┘
   │ ┌─────────────┐ │        
   │ │ RTCPeerConn │ │
   │ │ (信令)      │ │
   │ │ (signaling) │ │
   │ ├─────────────┤ │
   │ │ WebSocket   │ │
   │ │ (SDP/ICE)   │ │
   │ ├─────────────┤ │
   │ │ <video>     │ │
   │ │ (播放)      │ │
   │ │ (playback)  │ │
   │ └─────────────┘ │
   └─────────────────┘
```

### 2. 流程摘要
### 2. Flow summary

1. `VisionPilot` 应用在其主循环中调用 `webrtc_streamer->push_frame(cv::Mat)`。该帧由 `visualization::render_frame()` 生成。
1. `VisionPilot` application calls `webrtc_streamer->push_frame(cv::Mat)` in its main loop. This frame is generated from `visualization::render_frame()`.
2. 帧经过验证，转换为 BGR，并推送到 GStreamer 流水线的 `appsrc` 元素。
2. Frame is validated, converted to BGR, and pushed to the GStreamer pipeline's `appsrc` element.
3. GStreamer 使用 VP8 编解码器对帧进行编码，并将其馈入 `webrtcbin` 元素。
3. GStreamer encodes the frame using VP8 codec and feeds it to the `webrtcbin` element.
4. 在第一帧时，`webrtcbin` 触发 `on-negotiation-needed`，创建 SDP 提供。
4. On the first frame, `webrtcbin` triggers `on-negotiation-needed`, which creates an SDP offer.
5. SDP 提供被排队并通过 WebSocket 发送到浏览器客户端。
5. SDP offer is queued and sent to the browser client via WebSocket.
6. 浏览器回复 SDP 应答和 ICE 候选。
6. Browser responds with SDP answer and ICE candidates.
7. 服务器接收应答，设置远程描述，并刷新任何待处理的 ICE 候选。
7. Server receives answer, sets remote description, and flushes any pending ICE candidates.
8. 媒体流开始通过已建立的对等连接从服务器流向浏览器。
8. Media stream begins flowing from server to browser via the established peer connection.

### 3. 模块结构
### 3. Module structure

```
visualization/
├── CMakeLists.txt
├── README.md (本文件 / this file)
├── include/
│   └── visualization/
│       ├── visualization.hpp           (可视化头文件 / visualization header)
│       └── visualization_to_webrtc.hpp (WebRTC 头文件 / WebRTC header)
└── src/
    ├── visualization.cpp               (可视化绘图，OpenCV 窗口管理 / visualization drawing, OpenCV window management)
    └── visualization_to_webrtc.cpp     (WebRTC 实现 / WebRTC implementation)
```

## III. 构建
## III. Build

### 1. 前置条件
### 1. Prerequisites

- `ROS2 Humble`（在 Ubuntu 22.04 上测试）/ `ROS2 Humble` (tested on Ubuntu 22.04)
    - `source /opt/ros/humble/setup.bash`
- `GStreamer` 开发库：/ `GStreamer` development libraries:
    - `libgstreamer1.0-dev`
    - `libgstreamer-plugins-base1.0-dev`
    - `libgstreamer-plugins-bad1.0-dev`
- `libsoup 2.4`（HTTP/WebSocket 服务器）：/ `libsoup 2.4` (HTTP/WebSocket server):
    - `libsoup2.4-dev`
- `JSON-GLib`（JSON 信令消息处理）：/ `JSON-GLib` (JSON signaling message handling):
    - `libjson-glib-dev`
- `OpenCV`:
  - `libopencv-dev`
- `标准构建工具`：/ `Standard build tools`:
  - `build-essential`, `cmake`（≥3.22.1）, `pkg-config`

一次性安装所有：
Install all at once:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config \
  libopencv-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-bad1.0-dev \
  libsoup2.4-dev libjson-glib-dev
```

### 2. 步骤
### 2. Steps

```bash
# 1. 导航到工作区根目录 / Navigate to workspace root
cd /path/to/VisionPilot/development_releases/1.0

# 2. 加载 ROS2 / Source ROS2
source /opt/ros/humble/setup.bash

# 3. 构建（从工作区根目录；CMake 将配置所有模块）
# 3. Build (from workspace root; CMake will configure all modules)
mkdir -p build && cd build
cmake .. -DONNXRUNTIME_ROOT=$your_ONNXRUNTIME_path
make -j$(nproc)
```

### 3. 预期输出
### 3. Expected Output

```bash
[ 83%] Built target visualization
[ 89%] Building CXX object app/CMakeFiles/VisionPilot.dir/vision_pilot.cpp.o
[ 97%] Linking CXX executable ../VisionPilot
[100%] Built target VisionPilot
```

二进制位置：`build/VisionPilot`
Binary location: `build/VisionPilot`

## IV. 测试/演示
## IV. Test/demo

### 1. 启用 WebRTC 运行
### 1. Running with WebRTC enabled

本演示将指导你测试 WebRTC 流式传输，使用 V4L2 挂载，通过 `v4l2loopback` 内核模块和 FFmpeg 的组合进行流式传输。
This demo shall guide you through testing this WebRTC streaming with a V4L2 mount, streamed via a combination of the `v4l2loopback` kernel module and FFmpeg.

通过本演示，你将：
With this demo, you will:
1. 从本地视频发布 V4L2 视频流挂载。
1. Publish a V4L2 video streaming mount from a local video.
2. 使用 `VisionPilot` 应用订阅该流挂载，处理帧并流式传输到本地主机。
2. Use `VisionPilot` application to subscribe to that streaming mount, process and stream frames to a local host.

```bash
# 1. 导航到构建目录 / Navigate to build directory
cd /path/to/VisionPilot/development_releases/1.0/build

# 2. 初始化 V4L2 流挂载 / Initiate V4L2 streaming mount

# a. 安装包 / Install package
sudo apt update
sudo apt install ffmpeg -y
sudo apt install v4l2loopback-dkms -y

# b. 加载模块（假设你将在 `/dev/video9` 流式传输）
# b. Load the module (assuming you gonna stream it at `/dev/video9`)
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=9 card_label="Virtual Camera" exclusive_caps=1

# c. 在该挂载发布循环视频 / Publish looping video at that mount
ffmpeg -re -stream_loop -1 -i <absolute path to local video> -f v4l2 -pix_fmt yuv420p /dev/video9

# 3. 启动 VisionPilot 应用，V4L2 订阅到该挂载，并流式传输到 http://127.0.0.1:8080/
# 3. Kickstart VisionPilot app with V4L2 subscription to that mount, and stream frames to http://127.0.0.1:8080/
./VisionPilot 1 /dev/video0 10 1 8080
```

**参数：**
**Arguments:**

- `1`：V4L2 模式（使用 `0` 表示 ROS2 模式）/ V4L2 mode (use `0` for ROS2 mode)
- `/dev/video0`：V4L2 设备路径（如果是 ROS2 模式，第二个参数将是 ROS2 话题名称）/ V4L2 device path (if ROS2 mode, this second arg will be ROS2 topic name)
- `10`：目标 FPS / target FPS
- `1`：启用 WebRTC（使用 `0` 禁用）/ enable WebRTC (use `0` to disable)
- `8080`：WebRTC 服务器端口（如果 WebRTC 禁用则不可用）/ WebRTC server port (not available if WebRTC is disabled)

**预期终端输出：**
**Expected terminal output:**

```bash
Starting in V4L2 mode with device: /dev/video9 and FPS: 10
[V4L2Reader INFO] Initializing V4L2 Reader
[V4L2Reader INFO]   Device Path: /dev/video9
[V4L2Reader INFO]   Target FPS: 10
[V4L2Reader INFO] V4L2 device configured successfully
[V4L2Reader INFO]   Received resolution: 2560x1440
[V4L2Reader INFO]   Received FPS: 10.000000
Starting WebRTC streamer on port: 8080
[WebRTCStreamer] soup_server created
[WebRTCStreamer] soup_server listening on port 8080 and handlers installed
[WebRTCStreamer] pipeline created, appsrc=0x57add3f02e20 webrtc=0x57add3f34110
[WebRTCStreamer] pipeline set to PLAYING
Open browser at: http://127.0.0.1:8080/
Local OpenCV preview is disabled while WebRTC is enabled.
```

### 2. 访问流
### 2. Accessing the stream

1. 打开网络浏览器并导航到：`http://127.0.0.1:8080/`
1. Open a web browser and navigate to: `http://127.0.0.1:8080/`
2. 最小化 HTML 客户端页面将加载。
2. The minimal HTML client page will load.
3. WebRTC 协商将自动开始。
3. WebRTC negotiation will begin automatically.
4. 提供/应答交换完成后，视频流将出现在 `<video>` 元素中。
4. Once offer/answer exchange completes, the video stream should appear in the `<video>` element.

### 3. 故障排除
### 3. Troubleshooting

以下是已知/遇到的 bug 和错误。如果你遇到全新的问题，请尝试在 [Autoware VisionPilot 仓库](https://github.com/autowarefoundation/autoware_vision_pilot)中发布新 issue。
These are known/enountered bugs and errors. If you encounter a completely new one, try posting it as new issue at [Autoware VisionPilot repository](https://github.com/autowarefoundation/autoware_vision_pilot).

1. **黑屏/空白视频**：帧可能未从相机到达。在禁用 WebRTC 的情况下测试：
1. **Black/blank video**: frame may not be arriving from the camera. Test with WebRTC disabled:
   
    ```bash
    ./VisionPilot 1 /dev/video0 10 0
    ```
    
    如果 OpenCV 预览正常，相机没问题，那么问题可能是 WebRTC 特有的。
    If OpenCV preview works, the camera is fine, then the issue might be somewhat WebRTC-specific.
  
2. **连接被拒绝**：确保端口 `8080` 未被占用：
2. **Connection refused**: ensure port `8080` is not in use:
  
    ```bash
    lsof -i :8080
    ```

## V. 技术细节
## V. Technical details

### 1. 核心组件与数据流
### 1. Core components & data flow

#### a. WebRTCStreamer 公共接口
#### a. WebRTCStreamer public interface

**位置：** `include/visualization/visualization_to_webrtc.hpp`
**Location:** `include/visualization/visualization_to_webrtc.hpp`

`WebRTCStreamer` 类提供清晰的公共 API：
The `WebRTCStreamer` class provides a clean public API:

```cpp
class WebRTCStreamer {
    struct Config { ... };                  // 配置：主机、端口、路径、帧率 / Configuration: host, port, path, frame_rate
    WebRTCStreamer(Config config);          // 构造函数 / Constructor
    ~WebRTCStreamer();                      // 析构函数 / Destructor
    
    bool start();                           // 初始化服务器、GStreamer、事件循环 / Initialize server, GStreamer, event loop
    bool stop();                            // 清理和关闭 / Cleanup and shutdown
    bool push_frame(const cv::Mat& frame);  // 提交帧用于编码/流式传输 / Submit frame for encoding/streaming
    bool is_running() const;                // 检查服务器是否活跃 / Check if server is active
    bool has_client() const;                // 检查浏览器是否已连接 / Check if browser is connected
    std::string browser_url() const;        // 获取浏览器连接的 URL / Get URL for browser to connect
    
private:
    std::unique_ptr<Impl> impl;             // 私有实现模式 / Private implementation pattern
};
```

上面的私有实现 `impl` 用于将公共接口与内部复杂性分离。
Above private implementation `impl` is used to separate public interface from internal complexity.

#### b. WebRTCStreamer::Impl 结构
#### b. WebRTCStreamer::Impl structure

**位置：** `src/visualization_to_webrtc.cpp`（约第 196-250 行）
**Location:** `src/visualization_to_webrtc.cpp` (lines ~196–250)

`Impl` 结构封装了所有内部状态和逻辑：
The `Impl` struct encapsulates all internal state and logic:

```cpp
struct WebRTCStreamer::Impl {

    // ===== 配置 / CONFIGURATION =====
    Config config;                              // 用户提供的设置 / User-provided settings
    
    // ===== 信令（HTTP + WebSocket）/ SIGNALING (HTTP + WebSocket) =====
    SoupServer *server;                         // libsoup HTTP 服务器 / libsoup HTTP server
    GMainLoop *main_loop;                       // GLib 事件循环 / GLib event loop
    std::thread server_thread;                  // 运行事件循环的线程 / Thread running the event loop
    
    // ===== GSTREAMER 流水线 / GSTREAMER PIPELINE =====
    GstElement *pipeline;                       // 顶层流水线元素 / Top-level pipeline element
    GstElement *appsrc;                         // 输入：从应用接收帧 / Input: receives frames from app
    GstElement *webrtc;                         // webrtcbin：处理 RTC 逻辑 / webrtcbin: handles RTC logic
    
    // ===== 信令状态 / SIGNALING STATE =====
    mutable std::mutex signal_mutex;
    SoupWebsocketConnection *client_connection; // 到浏览器的活跃 WS 连接 / Active WS connection to browser
    std::vector<std::string> pending_signals;   // 排队的 SDP/ICE 消息 / Queued SDP/ICE messages
    
    // ===== 远程描述与 ICE 状态 / REMOTE DESCRIPTION & ICE STATE =====
    std::mutex remote_candidate_mutex;
    std::vector<std::pair<int, std::string>> pending_remote_candidates;
    std::atomic<bool> remote_description_ready; // 标志：可以添加 ICE 候选？/ Flag: can add ICE candidates?
    
    // ===== 帧流式传输状态 / FRAME STREAMING STATE =====
    std::atomic<bool> running;                  // 流水线活跃？/ Pipeline active?
    std::atomic<uint64_t> frame_index;          // 用于 PTS 的单调帧计数器 / Monotonic frame counter for PTS
    bool caps_configured;                       // GStreamer caps 已设置？/ GStreamer caps set?
    int configured_width, configured_height;    // 上次配置的帧尺寸 / Last configured frame dimensions
};
```

**关键设计决策：**
**Key design decisions:**

- **signal_mutex** 保护对 `client_connection` 和 `pending_signals` 的并发访问（应用线程推送帧；服务器线程发送信令）。
- **Mutex for signal_mutex** protects concurrent access to `client_connection` and `pending_signals` (app thread pushes frames; server thread sends signals).
- **原子标志**（`running`、`remote_description_ready`、`frame_index`）为频繁检查的状态提供无锁读/写。
- **Atomic flags** (`running`, `remote_description_ready`, `frame_index`) provide lock-free read/write for frequently checked state.
- **待处理队列**将帧推送（应用线程）与信令（服务器线程）解耦。
- **Pending queues** decouple frame pushing (app thread) from signaling (server thread).

#### c. 帧时间戳与同步
#### c. Frame timestamping & synchronization

在 `push_frame()` 中（约第 770-820 行），每帧都带有时间戳：
In `push_frame()` (lines ~770–820), each frame is timestamped:

```cpp
const guint64 duration_ns = config.frame_rate > 0.0
    ? static_cast<guint64>(GST_SECOND / config.frame_rate)
    : GST_CLOCK_TIME_NONE;

const guint64 pts_ns = (
    duration_ns == GST_CLOCK_TIME_NONE ? 0 : 
    frame_index.fetch_add(1, std::memory_order_acq_rel) * duration_ns
);

GST_BUFFER_PTS(buffer) = pts_ns;
GST_BUFFER_DTS(buffer) = pts_ns;
GST_BUFFER_DURATION(buffer) = duration_ns;
```

解释：
Explanation:

- **PTS（呈现时间戳）** 告诉解码器何时显示帧。
- **PTS (Presentation Time Stamp)** tells decoder when to display the frame.
- **DTS（解码时间戳）** 告诉解码器何时解码帧。
- **DTS (Decoding Time Stamp)** tells decoder when to decode the frame.
- **Duration** 是帧显示持续时间（帧率的倒数）。
- **Duration** is frame display duration (inverse of framerate).
- **单调 `frame_index`** 确保 PTS 递增，防止抖动。
- **Monotonic `frame_index`** ensures PTS increases, preventing jitter.

例如，10 FPS = 每帧 100 ms：
For example, at 10 FPS = 100 ms per frame:

- 帧 0：PTS = 0 ns / Frame 0: PTS = 0 ns
- 帧 1：PTS = 100,000,000 ns（100 ms）/ Frame 1: PTS = 100,000,000 ns (100 ms)
- 帧 2：PTS = 200,000,000 ns（200 ms）/ Frame 2: PTS = 200,000,000 ns (200 ms)
- 依此类推 / etc.

这允许浏览器以正确的速度播放帧。
This allows the browser to play frames at the correct speed.

---

#### d. 格式验证 `ensure_bgr_frame()`
#### d. Format validation `ensure_bgr_frame()`

在匿名命名空间中（约第 148-190 行），帧被转换为 BGR：
In the anonymous namespace (lines ~148–190), frames are converted to BGR:

```cpp
cv::Mat ensure_bgr_frame(const cv::Mat& frame) {
    if (frame.empty()) return frame;
    if (frame.type() == CV_8UC3) return frame.isContinuous() ? frame : frame.clone();
    
    cv::Mat converted;
    if (frame.type() == CV_8UC1)        // 灰度 / Grayscale
        cv::cvtColor(frame, converted, cv::COLOR_GRAY2BGR);
    else if (frame.type() == CV_8UC4)   // BGRA
        cv::cvtColor(frame, converted, cv::COLOR_BGRA2BGR);
    else                                 // 未知 / Unknown
        frame.convertTo(converted, CV_8UC3);
    
    return converted;
}
```

由于 GStreamer 期望 `videoconvert` 元素使用 BGR，因此需要确保这一点。
Since GStreamer expects BGR for the `videoconvert` element, I shall ensure that.

---

### 2. WebRTC 信令流
### 2. WebRTC signaling flow

#### a. SDP 提供/应答交换
#### a. SDP offer/answer exchange

##### i. 序列（约第 461-510 行和第 768-820 行）
##### i. Sequence (lines ~461–510 and ~768–820)

1. 在第一帧推送时（约第 768 行）：
1. On first frame push (line ~768):
   - `webrtcbin` 元素检测到媒体流并发出 `on-negotiation-needed` 信号。
   - `webrtcbin` element detects media stream and emits `on-negotiation-needed` signal.
   
2. 在 `on_negotiation_needed()` 回调中（约第 518-530 行）：
2. Upon `on_negotiation_needed()` callback (lines ~518–530):
   ```cpp
   g_signal_emit_by_name(impl->webrtc, "create-offer", nullptr, promise);
   ```
   - 创建 GStreamer promise 以异步生成 SDP 提供。
   - Creates a GStreamer promise to generate an SDP offer asynchronously.
   
3. 在 `on_offer_created()` 回调中（约第 475-510 行）：
3. Upon `on_offer_created()` callback (lines ~475–510):
   - 接收生成的 SDP 提供。
   - Receives the generated SDP offer.
   - 将其设置为本地描述：`g_signal_emit_by_name(impl->webrtc, "set-local-description", ...)`。
   - Sets it as the local description: `g_signal_emit_by_name(impl->webrtc, "set-local-description", ...)`.
   - 将提供排队发送到浏览器：`impl->queue_signal(make_offer_message(sdp_text))`。
   - Queues the offer to be sent to browser: `impl->queue_signal(make_offer_message(sdp_text))`.
   
4. 浏览器通过 WebSocket 接收提供（约第 346-408 行）：
4. Browser receives offer via WebSocket (lines ~346–408):
   - 在 JavaScript 中：`pc.setRemoteDescription({type: 'offer', sdp: ...})`。
   - In JavaScript: `pc.setRemoteDescription({type: 'offer', sdp: ...})`.
   - 浏览器的 RTCPeerConnection 生成应答。
   - Browser's RTCPeerConnection generates an answer.
   - 浏览器通过 WebSocket 发回应答。
   - Browser sends answer back via WebSocket.
   
5. 服务器接收应答（约第 389-402 行）：
5. Server receives answer (lines ~389–402):
   - 解析传入的 JSON：`handle_remote_description(impl, sdp_answer_text)`。
   - Parses incoming JSON: `handle_remote_description(impl, sdp_answer_text)`.
   - 创建 `GstWebRTCSessionDescription` 并设置：`g_signal_emit_by_name(impl->webrtc, "set-remote-description", ...)`。
   - Creates `GstWebRTCSessionDescription` and sets it: `g_signal_emit_by_name(impl->webrtc, "set-remote-description", ...)`.
   - 标记 `remote_description_ready = true` 并刷新待处理的 ICE 候选。
   - Marks `remote_description_ready = true` and flushes pending ICE candidates.

#### b. ICE 候选交换
#### b. ICE candidate exchange

##### i. 服务器端（约第 537-551 行）
##### i. Server-side (lines ~537–551)

当服务器的 `webrtcbin` 收集到 ICE 候选时：
When server's `webrtcbin` gathers an ICE candidate:
```cpp
void on_ice_candidate(GstElement *element, guint mline_index, gchar *candidate, gpointer user_data) {
    impl->queue_signal(make_candidate_message(mline_index, candidate));
}
```

候选作为 JSON 发送到浏览器：
Candidate is sent to browser as JSON:
```json
{ "type": "candidate", "sdpMLineIndex": 0, "candidate": "candidate:..." }
```

##### ii. 浏览器端（kBrowserHtml 中约第 65-89 行）：
##### ii. Browser-side (lines ~65–89 in kBrowserHtml):

```javascript
if (p.type === 'candidate') {
    const candidate = {
        candidate: p.candidate,
        sdpMLineIndex: p.sdpMLineIndex
    };
    if (!pc.remoteDescription) {
        pendingCandidates.push(candidate);  // 排队直到提供被设置 / Queue until offer is set
        return;
    }
    try {
        await pc.addIceCandidate(candidate);
    } catch (e) {
        console.error('Error adding ICE candidate:', e);
    }
}
```

候选被排队直到 `remoteDescription` 被设置，然后全部刷新。这防止了 ICE 候选在 SDP 提供之前到达的竞争条件。
Candidates are queued until `remoteDescription` is set, then all flushed. This prevents race conditions where ICE candidates arrive before SDP offer.

### 3. 信令基础设施：HTTP + WebSocket
### 3. Signaling infra: HTTP + WebSocket

#### a. HTTP 服务器（`libsoup`）
#### a. HTTP server (`libsoup`)

在 `start()` 中（约第 576-606 行）：
In `start()` (lines ~576–606):

```cpp
server = soup_server_new("server-header", "VisionPilot", NULL);
soup_server_listen_local(server, config.port, SOUP_SERVER_LISTEN_IPV4_ONLY, &listen_error);
soup_server_add_handler(server, "/", root_http_handler, this, nullptr);
soup_server_add_websocket_handler(server, config.websocket_path.c_str(), ..., websocket_handler, ...);
```

- `root_http_handler()`（约第 253-276 行）
- `root_http_handler()` (lines ~253–276)

    - 响应 `GET /` 与嵌入的 HTML 客户端（`kBrowserHtml`）。
    - Responds to `GET /` with the embedded HTML client (`kBrowserHtml`).
    - 作为静态内容服务器；最小开销。
    - Serves as a static content server; minimal overhead.

#### b. WebSocket 信令
#### b. WebSocket signaling

- `websocket_handler()`（约第 415-459 行）：
- `websocket_handler()` (lines ~415–459):
    - 当浏览器连接到 WebSocket 时触发。
    - Triggered when browser connects to Websocket.
    - 注册消息和关闭回调。
    - Registers message and close callbacks.
    - 设置 keepalive 间隔以检测死连接。
    - Sets keepalive interval to detect dead connections.

- `on_websocket_message()`（约第 346-408 行）：
- `on_websocket_message()` (lines ~346–408):
    - 解析传入的 JSON（SDP 应答或 ICE 候选）。
    - Parses incoming JSON (SDP answer or ICE candidate).
    - 分派到适当的处理器（`handle_remote_description()` 或 `handle_remote_candidate()`）。
    - Dispatches to appropriate handler (`handle_remote_description()` or `handle_remote_candidate()`).

- `on_websocket_closed()`（约第 283-296 行）：
- `on_websocket_closed()` (lines ~283–296):
    - 当浏览器断开连接时清理连接引用。
    - Cleans up connection reference when browser disconnects.

#### c. 消息队列与线程安全
#### c. Message queueing & thread safety

- `queue_signal()`（约第 861-884 行）：
- `queue_signal()` (lines ~861–884):
    ```cpp
    void queue_signal(const std::string& signal) {
        std::lock_guard<std::mutex> lock(signal_mutex);
        if (client_connection != nullptr && is_open(client_connection)) {
            soup_websocket_connection_send_text(client_connection, signal.c_str());
        } else {
            pending_signals.push_back(signal);
        }
    }
    ```

- `flush_pending_signals()`（约第 887-912 行）：
- `flush_pending_signals()` (lines ~887–912):
    - 在新客户端连接时调用。
    - Called when new client connects.
    - 发送所有排队的信号（例如在客户端准备好之前生成的 SDP 提供）。
    - Sends all queued signals (e.g., SDP offer generated before client was ready).
    - 即使时序宽松也允许优雅的握手。
    - Allows graceful handshake even if timing is loose.

### 4. 最小化 HTML5 浏览器客户端
### 4. Browser client with a minimal piece of HTML5

#### a. 结构
#### a. Structure

`kBrowserHtml` 常量（约第 27-95 行）
`kBrowserHtml` constant (lines ~27–95)

```html
<!doctype html>
<html>
  <head>
    <title>VisionPilot</title>
    <style>/* 全屏视频 / Full-screen video */</style>
  </head>
  <body>
    <video id="video" autoplay playsinline muted></video>
    <script>/* WebRTC 客户端逻辑 / WebRTC client logic */</script>
  </body>
</html>
```

#### b. 关键 JavaScript 元素
#### b. Key JavaScript elements

1. RTCPeerConnection（`pc`）/ RTCPeerConnection (`pc`)
   ```javascript
   const pc = new RTCPeerConnection();
   pc.ontrack = e => { video.srcObject = e.streams[0]; };
   ```
   - 创建对等连接。
   - Creates peer connection.
   - 在传入轨道（视频流）时，分配给 `<video>` 元素。
   - On incoming track (video stream), assigns to `<video>` element.

2. WebSocket（`ws`）/ WebSocket (`ws`)
   ```javascript
   const ws = new WebSocket(scheme + location.host + '/ws');
   ```
   - 连接到服务器的 `/ws` 端点。
   - Connects to server's `/ws` endpoint.
   - 双向信令。
   - Bidirectional signaling.

3. 提供/应答处理 / Offer/Answer handling
   ```javascript
   if (p.type === 'offer') {
       await pc.setRemoteDescription({type: 'offer', sdp: p.sdp});
       const a = await pc.createAnswer();
       await pc.setLocalDescription(a);
       ws.send(JSON.stringify({type: 'answer', sdp: pc.localDescription.sdp}));
   }
   ```

4. ICE 候选排队 / ICE candidate queueing
   ```javascript
   async function drainPendingCandidates() {
       while (pendingCandidates.length > 0) {
           const c = pendingCandidates.shift();
           await pc.addIceCandidate(c);
       }
   }
   ```
   - 确保候选仅在远程描述设置后添加。
   - Ensures candidates are added only after remote description is set.

### 5. 线程模型
### 5. Threading model

#### a. 线程流
#### a. Threading flow

```
┌─ 应用线程（主线程）
│  Application thread (main)
│  │
│  ├─ 调用：webrtc_streamer.push_frame() [=> impl->push_frame()]
│  │  Calls: webrtc_streamer.push_frame() [=> impl->push_frame()]
│  │          │
│  │          ├─ 验证帧 / Validates frame
│  │          ├─ 调用：gst_app_src_push_buffer()
│  │          │  (线程安全；入队到流水线的 queue 元素)
│  │          │  (thread-safe; enqueues to pipeline's queue element)
│  │          └─ 返回 / Returns
│  │
│  └─ (继续应用循环，休眠 33ms)
│     (Continues app loop, sleeps 33ms)
│
├─ GStreamer 流水线线程（由 g_main_loop_run 生成）
│  GStreamer pipeline thread (spawned by g_main_loop_run)
│  │
│  ├─ 从 appsrc 接收缓冲区 / Receives buffers from appsrc
│  ├─ 通过流水线处理：queue => videoconvert => vp8enc => webrtcbin
│  │  Processes through pipeline: queue => videoconvert => vp8enc => webrtcbin
│  ├─ 发出信号：on-negotiation-needed, on-ice-candidate
│  │  Emits signals: on-negotiation-needed, on-ice-candidate
│  ├─ 回调在此线程上下文中间步调用
│  │  Callbacks invoked synchronously in this thread context
│  └─ (运行直到流水线状态 => NULL)
│     (Runs until pipeline state => NULL)
│
└─ libsoup 事件循环线程（在 start() 中生成）
   libsoup event loop thread (spawned in start())
   │
   ├─ g_main_loop_run() 处理 I/O 事件
   │  g_main_loop_run() processes I/O events
   ├─ 处理传入的 HTTP GET / WebSocket 消息
   │  Handles incoming HTTP GET / WebSocket messages
   ├─ 调用：root_http_handler, websocket_handler, on_websocket_message
│  Calls: root_http_handler, websocket_handler, on_websocket_message
   ├─ 访问 impl->client_connection（由 signal_mutex 保护）
   │  Accesses impl->client_connection (protected by signal_mutex)
   └─ (运行直到 g_main_loop_quit())
      (Runs until g_main_loop_quit())
```

#### b. 并发性
#### b. Concurrency

- 推送帧（应用线程）=> GStreamer 队列（使用 GStreamer 的内部队列，线程安全）。
- Push frame (app thread) => Queue in GStreamer (thread-safe with GStreamer's internal queues).
- ICE 候选发射（GStreamer 线程）=> 队列信号（由 `signal_mutex` 保护）。
- ICE candidate emission (GStreamer thread) => Queue signal (protected by `signal_mutex`).
- WebSocket 接收（libsoup 线程）=> 更新状态（由 `signal_mutex`、`remote_candidate_mutex` 保护）。
- WebSocket receive (libsoup thread) => Update state (protected by `signal_mutex`, `remote_candidate_mutex`).

### 6. 配置
### 6. Configuration

`Config` 结构允许自定义：
The `Config` struct allows customization:

```cpp
struct Config {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    std::string websocket_path = "/ws";
    double frame_rate = 10.0;
};
```
