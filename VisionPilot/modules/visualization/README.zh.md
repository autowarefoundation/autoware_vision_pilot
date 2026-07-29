# 可视化模块

[🇺🇸 English](README.md)

## 致谢

感谢 [Ethan](https://dev.to/ethand91) 及其博客文章 [使用 C++ WebRTC GStreamer 流式传输相机](https://dev.to/ethand91/streaming-camera-with-c-webrtc-gstreamer-pof)。

你的实现对我完成本模块确实非常有帮助和启发。

## I. 概述

WebRTC 可视化模块通过 WebRTC 协议为 VisionPilot 流水线提供实时视频流能力。它提供以下核心功能：

1. **实时帧捕获和编码**：接受 OpenCV `cv::Mat` 帧并通过 GStreamer 将其编码为 VP8 视频编解码器。
2. **WebRTC 点对点流式传输**：在服务器（VisionPilot 应用）和浏览器客户端之间建立 WebRTC 对等连接，实现通过互联网或局域网的实时视频传输。
3. **轻量级浏览器客户端**：提供一个最小化的自包含 HTML5 页面，内置 WebRTC JavaScript 客户端，浏览器无需外部依赖。
4. 实现基于 WebSocket 的信令，用于 SDP（会话描述协议）提供/应答协商和 ICE（交互式连接建立）候选交换。
5. **线程安全的帧流式传输**：管理来自主应用线程的并发帧推送，同时在单独的线程中运行 GStreamer 流水线和事件循环。

本模块对于开发和测试阶段的自主驾驶流水线的远程监控、调试和可视化至关重要。

## II. 架构与模块结构

### 1. 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    VisionPilot 应用                              │
│                  (vision_pilot.cpp 主线程)                       │
│                                                                 │
│   数据采集                                                       │
│   ┌──────────────────────┐                                      │
│   │  V4L2/ROS2 相机       │                                      │
│   │      源               │                                      │
│   └──────────┬───────────┘                                      │
│              │                                                  │
│              │ cv::Mat 帧 (33ms 循环)                            │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────────────────────────┐                  │
│   │  (各种上游模块，如                        │                  │
│   │  模型推理、处理、计算等)                   │                  │
│   └──────────┬───────────────────────────────┘                  │
│              │                                                  │
│              │ 帧 + 纵向/横向规划结果                            │
│              │                                                  │
│              ▼                                                  │
│   可视化                                                         │
│   ┌──────────────────────────────────────────┐                  │
│   │  visualization::render_frame()           │                  │
│   │  (绘制帧 + 规划结果)                      │                  │
│   └──────────┬───────────────────────────────┘                  │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────────────────────────┐                  │
│   │  WebRTCStreamer::push_frame()            │                  │
│   │  (通过 WebRTC 流式传输到端点)             │                  │
│   └──────────┬───────────────────────────────┘                  │
│              │                                                  │
└──────────────┼──────────────────────────────────────────────────┘
               │
               │ BGR 帧 + 元数据
               │
    ┌──────────▼───────────────────────────────────────────┐
    │         WebRTCStreamer::Impl（内部实现）              │
    │                                                      │
    │  ┌────────────────────────────────────────────────┐  │
    │  │  GStreamer 流水线（单独线程）                    │  │
    │  │                                                │  │
    │  │  appsrc => queue => videoconvert => vp8enc =>  │  │
    │  │  rtpvp8pay => webrtcbin                        │  │
    │  │                                                │  │
    │  │  ┌──────────────────────────────────────────┐  │  │
    │  │  │ WebRTC 对等连接（GStreamer）              │  │  │
    │  │  │  - 管理媒体流                            │  │  │
    │  │  │  - 生成 SDP 提供                         │  │  │
    │  │  │  - 收集 ICE 候选                         │  │  │
    │  │  └──────────────────────────────────────────┘  │  │
    │  └────────────────────────────────────────────────┘  │
    │                                                      │
    │  ┌────────────────────────────────────────────────┐  │
    │  │  信令层（SoupServer + WebSocket）               │  │
    │  │                                                │  │
    │  │  HTTP 处理器：                                  │  │
    │  │    GET / => 提供 kBrowserHtml                  │  │
    │  │                                                │  │
    │  │  WebSocket 处理器：                             │  │
    │  │    - 接收：SDP 应答、ICE 候选                   │  │
    │  │    - 发送：SDP 提供、ICE 候选                   │  │
    │  │    - 队列 + 刷新机制用于排序                    │  │
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
   │  (HTML5 + JS)   │ <====> │  (Internet/LAN)  │
   │                 │        └──────────────────┘
   │ ┌─────────────┐ │        
   │ │ RTCPeerConn │ │
   │ │ (信令)      │ │
   │ ├─────────────┤ │
   │ │ WebSocket   │ │
   │ │ (SDP/ICE)   │ │
   │ ├─────────────┤ │
   │ │ <video>     │ │
   │ │ (播放)      │ │
   │ └─────────────┘ │
   └─────────────────┘
```

### 2. 流程摘要

1. `VisionPilot` 应用在其主循环中调用 `webrtc_streamer->push_frame(cv::Mat)`。该帧由 `visualization::render_frame()` 生成。
2. 帧经过验证，转换为 BGR，并推送到 GStreamer 流水线的 `appsrc` 元素。
3. GStreamer 使用 VP8 编解码器对帧进行编码，并将其馈入 `webrtcbin` 元素。
4. 在第一帧时，`webrtcbin` 触发 `on-negotiation-needed`，创建 SDP 提供。
5. SDP 提供被排队并通过 WebSocket 发送到浏览器客户端。
6. 浏览器回复 SDP 应答和 ICE 候选。
7. 服务器接收应答，设置远程描述，并刷新任何待处理的 ICE 候选。
8. 媒体流开始通过已建立的对等连接从服务器流向浏览器。

### 3. 模块结构

```
visualization/
├── CMakeLists.txt
├── README.md (本文件)
├── include/
│   └── visualization/
│       ├── visualization.hpp           (可视化头文件)
│       └── visualization_to_webrtc.hpp (WebRTC 头文件)
└── src/
    ├── visualization.cpp               (可视化绘图，OpenCV 窗口管理)
    └── visualization_to_webrtc.cpp     (WebRTC 实现)
```

## III. 构建

### 1. 前置条件

- `ROS2 Humble`（在 Ubuntu 22.04 上测试）
    - `source /opt/ros/humble/setup.bash`
- `GStreamer` 开发库：
    - `libgstreamer1.0-dev`
    - `libgstreamer-plugins-base1.0-dev`
    - `libgstreamer-plugins-bad1.0-dev`
- `libsoup 2.4`（HTTP/WebSocket 服务器）：
    - `libsoup2.4-dev`
- `JSON-GLib`（JSON 信令消息处理）：
    - `libjson-glib-dev`
- `OpenCV`:
  - `libopencv-dev`
- `标准构建工具`：
  - `build-essential`, `cmake`（≥3.22.1）, `pkg-config`

一次性安装所有：

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

```bash
# 1. 导航到工作区根目录
cd /path/to/VisionPilot/development_releases/1.0

# 2. 加载 ROS2
source /opt/ros/humble/setup.bash

# 3. 构建（从工作区根目录；CMake 将配置所有模块）
mkdir -p build && cd build
cmake .. -DONNXRUNTIME_ROOT=$your_ONNXRUNTIME_path
make -j$(nproc)
```

### 3. 预期输出

```bash
[ 83%] Built target visualization
[ 89%] Building CXX object app/CMakeFiles/VisionPilot.dir/vision_pilot.cpp.o
[ 97%] Linking CXX executable ../VisionPilot
[100%] Built target VisionPilot
```

二进制位置：`build/VisionPilot`

## IV. 测试/演示

### 1. 启用 WebRTC 运行

本演示将指导你测试 WebRTC 流式传输，使用 V4L2 挂载，通过 `v4l2loopback` 内核模块和 FFmpeg 的组合进行流式传输。

通过本演示，你将：
1. 从本地视频发布 V4L2 视频流挂载。
2. 使用 `VisionPilot` 应用订阅该流挂载，处理帧并流式传输到本地主机。

```bash
# 1. 导航到构建目录
cd /path/to/VisionPilot/development_releases/1.0/build

# 2. 初始化 V4L2 流挂载

# a. 安装包
sudo apt update
sudo apt install ffmpeg -y
sudo apt install v4l2loopback-dkms -y

# b. 加载模块（假设你将在 `/dev/video9` 流式传输）
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback video_nr=9 card_label="Virtual Camera" exclusive_caps=1

# c. 在该挂载发布循环视频
ffmpeg -re -stream_loop -1 -i <absolute path to local video> -f v4l2 -pix_fmt yuv420p /dev/video9

# 3. 启动 VisionPilot 应用，V4L2 订阅到该挂载，并流式传输到 http://127.0.0.1:8080/
./VisionPilot 1 /dev/video0 10 1 8080
```

**参数：**

- `1`：V4L2 模式（使用 `0` 表示 ROS2 模式）
- `/dev/video0`：V4L2 设备路径（如果是 ROS2 模式，第二个参数将是 ROS2 话题名称）
- `10`：目标 FPS
- `1`：启用 WebRTC（使用 `0` 禁用）
- `8080`：WebRTC 服务器端口（如果 WebRTC 禁用则不可用）

**预期终端输出：**

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

1. 打开网络浏览器并导航到：`http://127.0.0.1:8080/`
2. 最小化 HTML 客户端页面将加载。
3. WebRTC 协商将自动开始。
4. 提供/应答交换完成后，视频流将出现在 `<video>` 元素中。

### 3. 故障排除

以下是已知/遇到的 bug 和错误。如果你遇到全新的问题，请尝试在 [Autoware VisionPilot 仓库](https://github.com/autowarefoundation/autoware_vision_pilot)中发布新 issue。

1. **黑屏/空白视频**：帧可能未从相机到达。在禁用 WebRTC 的情况下测试：
   
    ```bash
    ./VisionPilot 1 /dev/video0 10 0
    ```
    
    如果 OpenCV 预览正常，相机没问题，那么问题可能是 WebRTC 特有的。
  
2. **连接被拒绝**：确保端口 `8080` 未被占用：
   
    ```bash
    lsof -i :8080
    ```

## V. 技术细节

### 1. 核心组件与数据流

#### a. WebRTCStreamer 公共接口

**位置：** `include/visualization/visualization_to_webrtc.hpp`

`WebRTCStreamer` 类提供清晰的公共 API：

```cpp
class WebRTCStreamer {
    struct Config { ... };
    WebRTCStreamer(Config config);
    ~WebRTCStreamer();
    
    bool start();
    bool stop();
    bool push_frame(const cv::Mat& frame);
    bool is_running() const;
    bool has_client() const;
    std::string browser_url() const;
    
private:
    std::unique_ptr<Impl> impl;
};
```

上面的私有实现 `impl` 用于将公共接口与内部复杂性分离。

#### b. WebRTCStreamer::Impl 结构

**位置：** `src/visualization_to_webrtc.cpp`（约第 196-250 行）

`Impl` 结构封装了所有内部状态和逻辑：

```cpp
struct WebRTCStreamer::Impl {

    // ===== 配置 =====
    Config config;
    
    // ===== 信令（HTTP + WebSocket） =====
    SoupServer *server;
    GMainLoop *main_loop;
    std::thread server_thread;
    
    // ===== GSTREAMER 流水线 =====
    GstElement *pipeline;
    GstElement *appsrc;
    GstElement *webrtc;
    
    // ===== 信令状态 =====
    mutable std::mutex signal_mutex;
    SoupWebsocketConnection *client_connection;
    std::vector<std::string> pending_signals;
    
    // ===== 远程描述与 ICE 状态 =====
    std::mutex remote_candidate_mutex;
    std::vector<std::pair<int, std::string>> pending_remote_candidates;
    std::atomic<bool> remote_description_ready;
    
    // ===== 帧流式传输状态 =====
    std::atomic<bool> running;
    std::atomic<uint64_t> frame_index;
    bool caps_configured;
    int configured_width, configured_height;
};
```

**关键设计决策：**

- **signal_mutex 互斥锁** 保护对 `client_connection` 和 `pending_signals` 的并发访问（应用线程推送帧；服务器线程发送信令）。
- **原子标志**（`running`、`remote_description_ready`、`frame_index`）为频繁检查的状态提供无锁读/写。
- **待处理队列**将帧推送（应用线程）与信令（服务器线程）解耦。

#### c. 帧时间戳与同步

在 `push_frame()` 中（约第 770-820 行），每帧都带有时间戳：

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

- **PTS（呈现时间戳）** 告诉解码器何时显示帧。
- **DTS（解码时间戳）** 告诉解码器何时解码帧。
- **Duration** 是帧显示持续时间（帧率的倒数）。
- **单调 `frame_index`** 确保 PTS 递增，防止抖动。

例如，10 FPS = 每帧 100 ms：

- 帧 0：PTS = 0 ns
- 帧 1：PTS = 100,000,000 ns（100 ms）
- 帧 2：PTS = 200,000,000 ns（200 ms）
- 依此类推

这允许浏览器以正确的速度播放帧。

---

#### d. 格式验证 `ensure_bgr_frame()`

在匿名命名空间中（约第 148-190 行），帧被转换为 BGR：

```cpp
cv::Mat ensure_bgr_frame(const cv::Mat& frame) {
    if (frame.empty()) return frame;
    if (frame.type() == CV_8UC3) return frame.isContinuous() ? frame : frame.clone();
    
    cv::Mat converted;
    if (frame.type() == CV_8UC1)
        cv::cvtColor(frame, converted, cv::COLOR_GRAY2BGR);
    else if (frame.type() == CV_8UC4)
        cv::cvtColor(frame, converted, cv::COLOR_BGRA2BGR);
    else
        frame.convertTo(converted, CV_8UC3);
    
    return converted;
}
```

由于 GStreamer 期望 `videoconvert` 元素使用 BGR，因此需要确保这一点。

---

### 2. WebRTC 信令流

#### a. SDP 提供/应答交换

##### i. 序列（约第 461-510 行和第 768-820 行）

1. 在第一帧推送时（约第 768 行）：
   - `webrtcbin` 元素检测到媒体流并发出 `on-negotiation-needed` 信号。
    
2. 在 `on_negotiation_needed()` 回调中（约第 518-530 行）：
    ```cpp
    g_signal_emit_by_name(impl->webrtc, "create-offer", nullptr, promise);
    ```
    - 创建 GStreamer promise 以异步生成 SDP 提供。
    
3. 在 `on_offer_created()` 回调中（约第 475-510 行）：
    - 接收生成的 SDP 提供。
    - 将其设置为本地描述：`g_signal_emit_by_name(impl->webrtc, "set-local-description", ...)`。
    - 将提供排队发送到浏览器：`impl->queue_signal(make_offer_message(sdp_text))`。
    
4. 浏览器通过 WebSocket 接收提供（约第 346-408 行）：
    - 在 JavaScript 中：`pc.setRemoteDescription({type: 'offer', sdp: ...})`。
    - 浏览器的 RTCPeerConnection 生成应答。
    - 浏览器通过 WebSocket 发回应答。
    
5. 服务器接收应答（约第 389-402 行）：
    - 解析传入的 JSON：`handle_remote_description(impl, sdp_answer_text)`。
    - 创建 `GstWebRTCSessionDescription` 并设置：`g_signal_emit_by_name(impl->webrtc, "set-remote-description", ...)`。
    - 标记 `remote_description_ready = true` 并刷新待处理的 ICE 候选。

#### b. ICE 候选交换

##### i. 服务器端（约第 537-551 行）

当服务器的 `webrtcbin` 收集到 ICE 候选时：
```cpp
void on_ice_candidate(GstElement *element, guint mline_index, gchar *candidate, gpointer user_data) {
    impl->queue_signal(make_candidate_message(mline_index, candidate));
}
```

候选作为 JSON 发送到浏览器：
```json
{ "type": "candidate", "sdpMLineIndex": 0, "candidate": "candidate:..." }
```

##### ii. 浏览器端（kBrowserHtml 中约第 65-89 行）：

```javascript
if (p.type === 'candidate') {
    const candidate = {
        candidate: p.candidate,
        sdpMLineIndex: p.sdpMLineIndex
    };
    if (!pc.remoteDescription) {
        pendingCandidates.push(candidate);
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

### 3. 信令基础设施：HTTP + WebSocket

#### a. HTTP 服务器（`libsoup`）

在 `start()` 中（约第 576-606 行）：

```cpp
server = soup_server_new("server-header", "VisionPilot", NULL);
soup_server_listen_local(server, config.port, SOUP_SERVER_LISTEN_IPV4_ONLY, &listen_error);
soup_server_add_handler(server, "/", root_http_handler, this, nullptr);
soup_server_add_websocket_handler(server, config.websocket_path.c_str(), ..., websocket_handler, ...);
```

- `root_http_handler()`（约第 253-276 行）

    - 响应 `GET /` 与嵌入的 HTML 客户端（`kBrowserHtml`）。
    - 作为静态内容服务器；最小开销。

#### b. WebSocket 信令

- `websocket_handler()`（约第 415-459 行）：
    - 当浏览器连接到 WebSocket 时触发。
    - 注册消息和关闭回调。
    - 设置 keepalive 间隔以检测死连接。

- `on_websocket_message()`（约第 346-408 行）：
    - 解析传入的 JSON（SDP 应答或 ICE 候选）。
    - 分派到适当的处理器（`handle_remote_description()` 或 `handle_remote_candidate()`）。

- `on_websocket_closed()`（约第 283-296 行）：
    - 当浏览器断开连接时清理连接引用。

#### c. 消息队列与线程安全

- `queue_signal()`（约第 861-884 行）：
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
    - 在新客户端连接时调用。
    - 发送所有排队的信号（例如在客户端准备好之前生成的 SDP 提供）。
    - 即使时序宽松也允许优雅的握手。

### 4. 最小化 HTML5 浏览器客户端

#### a. 结构

`kBrowserHtml` 常量（约第 27-95 行）

```html
<!doctype html>
<html>
  <head>
    <title>VisionPilot</title>
    <style>/* 全屏视频 */</style>
  </head>
  <body>
    <video id="video" autoplay playsinline muted></video>
    <script>/* WebRTC 客户端逻辑 */</script>
  </body>
</html>
```

#### b. 关键 JavaScript 元素

1. RTCPeerConnection（`pc`）
    ```javascript
    const pc = new RTCPeerConnection();
    pc.ontrack = e => { video.srcObject = e.streams[0]; };
    ```
    - 创建对等连接。
    - 在传入轨道（视频流）时，分配给 `<video>` 元素。

2. WebSocket（`ws`）
    ```javascript
    const ws = new WebSocket(scheme + location.host + '/ws');
    ```
    - 连接到服务器的 `/ws` 端点。
    - 双向信令。

3. 提供/应答处理
    ```javascript
    if (p.type === 'offer') {
        await pc.setRemoteDescription({type: 'offer', sdp: p.sdp});
        const a = await pc.createAnswer();
        await pc.setLocalDescription(a);
        ws.send(JSON.stringify({type: 'answer', sdp: pc.localDescription.sdp}));
    }
    ```

4. ICE 候选排队
    ```javascript
    async function drainPendingCandidates() {
        while (pendingCandidates.length > 0) {
            const c = pendingCandidates.shift();
            await pc.addIceCandidate(c);
        }
    }
    ```
    - 确保候选仅在远程描述设置后添加。

### 5. 线程模型

#### a. 线程流

```
┌─ 应用线程（主线程）
│  │
│  ├─ 调用：webrtc_streamer.push_frame() [=> impl->push_frame()]
│  │          │
│  │          ├─ 验证帧
│  │          ├─ 调用：gst_app_src_push_buffer()
│  │          │  (线程安全；入队到流水线的 queue 元素)
│  │          └─ 返回
│  │
│  └─ (继续应用循环，休眠 33ms)
│
├─ GStreamer 流水线线程（由 g_main_loop_run 生成）
│  │
│  ├─ 从 appsrc 接收缓冲区
│  ├─ 通过流水线处理：queue => videoconvert => vp8enc => webrtcbin
│  ├─ 发出信号：on-negotiation-needed, on-ice-candidate
│  ├─ 回调在此线程上下文中间步调用
│  └─ (运行直到流水线状态 => NULL)
│
└─ libsoup 事件循环线程（在 start() 中生成）
   │
   ├─ g_main_loop_run() 处理 I/O 事件
   ├─ 处理传入的 HTTP GET / WebSocket 消息
   ├─ 调用：root_http_handler, websocket_handler, on_websocket_message
   ├─ 访问 impl->client_connection（由 signal_mutex 保护）
   └─ (运行直到 g_main_loop_quit())
```

#### b. 并发性

- 推送帧（应用线程）=> GStreamer 队列（使用 GStreamer 的内部队列，线程安全）。
- ICE 候选发射（GStreamer 线程）=> 队列信号（由 `signal_mutex` 保护）。
- WebSocket 接收（libsoup 线程）=> 更新状态（由 `signal_mutex`、`remote_candidate_mutex` 保护）。

### 6. 配置

`Config` 结构允许自定义：

```cpp
struct Config {
    std::string host = "127.0.0.1";
    uint16_t port = 8080;
    std::string websocket_path = "/ws";
    double frame_rate = 10.0;
};
```
