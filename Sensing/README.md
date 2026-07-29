# Vision Pilot 传感器指南
# Vision Pilot Sensing Guide

## 单摄像头 — 50-55 度水平视场角，2MP 分辨率
## Single Camera - 50 - 55 degree H-Fov and 2MP resolution

对于 Vision Pilot 的基线版本，我们建议使用具有 GMSL2 接口、2MP 分辨率、水平视场角在 50-55 度之间的车载级相机。视场角更宽的相机不适合，因为这类相机缺乏捕捉远距离场景的能力，而这对高速公路驾驶和 ADAS 安全功能是必需的。Vision Pilot 栈设计为以 10Hz 运行，虽然你可以在更高 FPS 下运行 — 我们推荐 10Hz。普通人类驾驶员的反应时间相当于 4Hz，F1 赛车手的平均反应时间为 8Hz，因此 Vision Pilot 在 10Hz 运行时提供超人反应速度。
For the baseline version of Vision Pilot, we recommend using an automotive grade camera with a GMSL2 interface, 2MP resolution and a horizontal field of view between 50 and 55 degrees. Cameras with wider field of view are not suitable since such cameras lack the ability to capture the scene at a longer range, required for highway driving and ADAS safety features. The Vision Pilot stack is designed to run at 10Hz, although you can run it at higher FPS - we recommend 10Hz. Average human drivers have a reaction time equivalent to 4Hz and F1-drivers have an average reaction time of 8Hz, so Vision Pilot provides super-human reaction speeds whilst operating at 10Hz.

### 安装相机
### Mounting the camera

![Camera Mounting Guide](camera_mounting_guide.png)

我们建议将相机安装在前挡风玻璃后面、后视镜下方 — 类似于汽车 OEM 安装原厂 ADAS 相机的方式。相机应以零侧倾角沿车辆中心线朝前安装，对于乘用车有 1-3 度的轻微向下俯仰角，对于穿梭车、公交车或卡车等较高车辆则有 10-15 度的较大向下俯仰角。
We recommend mounting the camera behind the front windscreen, underneath the rear-view mirror - similar to how stock ADAS cameras are mounted by automotive OEMs. The camera should be mounted with zero roll angle along the centreline of the vehicle facing forward, with a slight pitch angle downward of 1-3 degrees for passenger cars and a larger pitch down angle of between 10-15 degrees for taller vehicles such as shuttles, buses or trucks.

通常，车载级 GMSL 评估相机设计有螺丝孔，可用于将相机安装到车体框架上。我们建议使用 L 型支架，通过大多数相机中预设计的螺丝孔将支架固定在相机背面。
Typically, automotive GMSL evaluation cameras are designed with screw holes which can be used to mount the camera to the body frame. We recommend using an L-brack which affixes to the back face of the camera using the pre-designed screw-holes available in most cameras.

我们建议使用 [Pixelman 相机支架](https://www.amazon.com/Pixelman-Adhesive-2PCS-Universal-Windshield-Bracket/dp/B0C5XQ8ZX8) — 虽然它通常设计用于后挡风玻璃安装相机，但也可以轻松用于将相机安装到前挡风玻璃上。
We recommend using the [Pixelman camera mount](https://www.amazon.com/Pixelman-Adhesive-2PCS-Universal-Windshield-Bracket/dp/B0C5XQ8ZX8) - although it is typically designed for adding cameras to the rear windsheild, it can also easily be used to mount a camera to the front windshield.

要将你的车载级相机安装到 Pixelman 相机支架上，你需要使用一个 L 型支架，将其拧入车载级相机背面的安装孔中，然后拧入 Pixelman 支架的安装板上，如上方参考图所示。
To attach your automotive camera to the Pixelman camera mount, you will need to use an L-faced bracket which screws into the back face mounting holes of your automotive camera and then screws into the mounting plate of the Pixelman camera as per the above reference image.
