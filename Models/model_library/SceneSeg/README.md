## SceneSeg
Self-driving cars are usually trained to detect specific object types, such as cars, pedestrians, buses, etc. Such approaches are prone to failure cases when a self-driving car encounters an unusual object that it hasn't seen before, e.g. a rickshaw, or if a self-driving car encounters a strange presentation of a known object, e.g. a cyclist that has fallen over. In these scenarios, self-driving cars are unable to detect these critical objects leading to dangerous driving outcomes. To address this, we have developed SceneSeg, a neural network that is able to segment all important foreground objects, irrespective of what that object is. SceneSeg is able to implicitly learn the visual features of foreground objects such as cars, buses, vans, pedestrians, cyclists, animals, rickshaws, trucks and other similar objects, even though it has not been explicitly trained to detect these object types. SceneSeg is also able to detect objects that are outside of its training data, such as tyres rolling down a highway, or a loose trailer. SceneSeg can also detect objects in unusual presentations that it hasn't seen during training. SceneSeg performs robustly across challenging weather and lighting conditions, including during heavy rain, snow and low light driving. SceneSeg performs out of the box on roads across the world without any parameter tuning. SceneSeg provides self-driving cars with a core safety layer, helping to address 'long-tail' edge cases which plauge object-level detectors. SceneSeg is part of the [AutoSeg Foundation Model](../AutoSeg/README.md) which forms the core of the vision-pipeline of the [Autoware Privately Owned Vehicle Autonomous Highway Pilot System](..).

<img src="../../../Media/SceneSeg_GIF_Rain.gif" width="100%">

During training, SceneSeg estimates three semantic classes

- `Foreground Objects`
- `Background Elements`
- `Drivable Road Surface`

However, during inference, we only use the outputs from the **`Foreground Objects`** class.

## Watch the explainer video
Please click the video link to play - [***Video link***](https://drive.google.com/file/d/1riGlT3Ct-O1Y2C0DqxemwWS233dJrY7F/view?usp=sharing)

## SceneSeg model weights
### [Link to Download Pytorch Model Weights *.pth](https://drive.google.com/file/d/1vCZMdtd8ZbSyHn1LCZrbNKMK7PQvJHxj/view?usp=sharing)
### [Link to Download Traced Pytorch Model *.pt](https://drive.google.com/file/d/1G2pKrjEGLGY1ouQdNPh11N-5LlmDI7ES/view?usp=drive_link)
### [Link to Download ONNX FP32 Weights *.onnx](https://drive.google.com/file/d/1l-dniunvYyFKvLD7k16Png3AsVTuMl9f/view?usp=drive_link)
### [Link to Download ONNX INT8 Wights *.onnx](https://drive.google.com/file/d/1yAIsUukWKBsbIvoKVAabwwKPmSXRy24D/view?usp=drive_link)

## Performance Results
SceneSeg was trained on a diverse dataset comprised of multiple open-source datasets, including ACDC, MUSES, IDDAW, Mapillary Vistas and the Comma10K datset. These datasets provide challenging training data covering a wide range of countries, road types, lighting conditions and weather conditions. The BDD100K dataset was not used during training and served as a broad and diverse test set.

Mean Intersection Over Union (mIoU) scores are provided for both validation and test data. Validation results are provided for each of the datasets which comprise the complete validation set, alongside the results for the entire validation set, which are presented in the Cross Dataset column. Per-class mIoU scores are provided, alongside mIoU averaged across classes, as well as an Overall mIoU score which calculates the mIoU between the full multi-class prediction and multi-class ground truth.

### Validation Set Performance - mIoU Scores
|| Cross Dataset | Mapillary| MUSES | ACDC | IDDAW | Comma10K |
|--------|---------------|------------------|-------|------|-------|----------|
| Overall | **90.7** | 91.1 | 83.7 | 89.3 | 87.2 | **92.5** |
| Background Objects | **93.5** | 93.7 | 89.1 | 93.2 | 90.0 | **95.1** |
| Foreground Objects | **58.2** | **60.9** | 35.7 | 46.9 | 58.6 | 58.9 |
| Drivable Road Surface | **84.2** | 85.7 | 70.8 | 74.4 | 81.8 | **86.3** |
| Class Average | **78.6** | **80.1** | 65.2 | 71.5 | 76.8 | **80.1** |

### Test Set Performance - mIoU Scores
|| BDD100K |
|-|---------|
| Overall | **91.5** |
| Background Objects | **94.3** |
| Foreground Objects | **69.8** |
| Drivable Road Surface | **71.3** |
| Class Average | **78.5** |

### Inference Speed
Inference speed tests were performed on a laptop equipped with an RTX3060 Mobile Gaming GPU, and an AMD Ryzen 7 5800H CPU. The SceneSeg network comprises a total of 223.43 Billion Floating Point Operations.

#### FP32 Precision
At FP32 precision, SceneSeg achieved 18.1 Frames Per Second inference speed

#### FP16 Precision
At FP16 precision, SceneSeg achieved 26.7 Frames Per Second inference speed