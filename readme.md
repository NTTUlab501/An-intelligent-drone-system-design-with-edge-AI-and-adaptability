# Data Governance for Sustainable Smart City - SP3: An Intelligent Drone System Design with Edge AI and Adaptability
回應國家重要挑戰之人工智慧主題研究專案-永續智慧城市之資料治理 - 子計畫3: 具邊緣人工智慧與可適應性之智能無人機系統設計

Project website: https://www.twaicoe.org/design-of-integrated-command-and-control-center-for-sustainable-smart-cities

![image](https://github.com/NTTUlab501/An-intelligent-drone-system-design-with-edge-AI-and-adaptability/blob/master/Scenario_25.png)
![image](https://github.com/NTTUlab501/An-intelligent-drone-system-design-with-edge-AI-and-adaptability/blob/master/UAV_KV260_2575.png)

This work presents the ETAUS, an Edge and Trustworthy AI UAV System, as a mobile sensing platform for air quality monitoring. To meet the real-time processing demands and achieve adaptivity and scalability, ETAUS employs an FPGA device as the main hardware computing architecture rather than relying solely on a microprocessor or integrating with GPUs. ETAUS contains a neural engine that can execute our customized AI model for air quality index (AQI) level classification, a dual-CNN model for river pollution index (RPI) classification, and a pre-trained model for detecting objects containing private information. ETAUS also incorporates a de-identification process, cryptographic functions and protection matrices to safeguard information and individuals' privacy. Furthermore, cryptographic functions and protection matrices are implemented as reconfigurable modules, which can accelerate processing and protect data privacy and be reconfigured as needed.

## System Implementation
| Item  |   |
| ------------- | ------------- |
| FPGA Platform  |AMD-Xilinx Kria KV260 Vision AI Starter Kit  |
| OS  | Petalinux  |
| Neural Engine | Xilinx DPU |

## Edge AI Models
| Item  |   |
| ------------- | ------------- |
| Air pollution detection (AQI Level Classification) | Customized ResNet50 |
| Water quality monitoring (RPI Level Classification) | Dual CNN |
| Detection for Objects with Individuals' Privacy| YOLOv4 |


## Dataset
| Item  |  Source  |
| ------------- | ------------- |
| Environmental Information Open Platform | https://data.epa.gov.tw/paradigm |
| Air Pollution Image Dataset from India and Nepal | https://www.kaggle.com/ds/3152196 |
| Water Pollution Image Dataset | https://github.com/ccusp7/D07-2 |

