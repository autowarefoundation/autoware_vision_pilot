# Vision Pilot - ADAS and Self Driving

<p align="center">
    <picture>
        <source media="(prefers-color-scheme: dark)">
        <img src="./Media/VisionPilot_logo.png" alt="VisionPilot" width="100%">
    </picture>
</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Discord](https://img.shields.io/discord/953808765935816715?label=Autoware%20Discord)](https://discord.com/invite/Q94UsPvReQ)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/autowarefoundation/autoware.privately-owned-vehicles)
![GitHub Repo stars](https://img.shields.io/github/stars/autowarefoundation/autoware.privately-owned-vehicles)

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=whit)
![ROS](https://img.shields.io/badge/ROS-22314E?style=for-the-badge&logo=ROS&logoColor=whit)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/company/the-autoware-foundation)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@autowarefoundation)
[![Website](https://img.shields.io/badge/website-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://autoware.org/)
⭐ Star us on GitHub — your support motivates us a lot!

</div>

## Open-source highway autonomy system

<img src="/Media/hero_GIF.gif" width="100%">

This codebase contains **productionizable and safety certifiable code for an open-source highway autonomy** system that can power safe, SAE Level-4+, autonomous driving around the world. It is designed to be integrated with automotive OEMs and Tier-1 suppliers in series production privately owned passenger vehicles.

To learn more about how to participate in this project, please read the [onboarding guide](/ONBOARDING.md)

## End-to-End AI Architecture
We utilize an **End-to-End AI Architecture** in which each component technology of the highway pilot system is powered through neural-networks. Over the evolution of the project, we are introducing component-based/modular End-to-End AI, full monolithic End-to-End AI, as well as a new Hybrid End-to-End AI framework for optimal system explainability, introspection, verification and safety validation.

## No Reliance on High Definition Maps
**We do not require 3D high definition  maps**. Our stack can operate in a 'mapless' mode or use existing 2D navigational (sat-nav style) maps, also called ADAS maps. Human beings can drive without prior knowledge of the detailed 3D geometry of roadways by relying on real-time scene perception, and our system mimics this process.

## Technology Roadmap
We are developing true hands-off, eyes-off autonomous driving which is enabled on highways across the world where no human supervision is required of the vehicle. Rather than going for a single moonshot, we are sequentially building the core software stack of the highway pilot system in a modular manner, and releasing versions of the autonomous highway pilot system that serve a core customer need, where each new release expands upon the operational-design-domain, technology prowess and safety envelope of the prior release.

As part of our technology roadmap, we will sequentially develop three versions of the Autoware Autonomous Highway Pilot System, called:
- ***[Vision Pilot - Level 2+ highway autopilot through component-based End-to-End AI](/VisionPilot/)***
- ***Vision Pilot PRO - Level 2+ highway and urban autopilot through monolithic End-to-End AI***
- ***Vision Drive - Level 4+ all road autonomy using hybrid End-to-End AI***

<img src="/Media/VisionPilot.png" width="100%">

<img src="/Media/VisionPilot_PRO.png" width="100%">

<img src="/Media/VisionDrive.png" width="100%">

## Models
All of our AI models are available under Apache 2.0 licence and can freely be used for commercial and research purposes. We release model architecture, data parsing scripts, training pipelines and model weights and fully open-source to help democratize access to this vital technology.

## Modules
Specific code modules for features such as Software-Defined vehicle implementaiton, Simulation, Control etc. can be found in the Modules folder.