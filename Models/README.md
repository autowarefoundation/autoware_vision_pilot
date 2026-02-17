![SceneSeg GIF](../Media/SceneSeg_GIF.gif) ![Scene3D GIF](../Media/Scene3D_GIF.gif) ![DomainSeg GIF 2](../Media/DomainSeg_GIF_2.gif) ![AutoSpeed GIF](../Media/AutoSpeed_GIF.gif) ![EgoLanes GIF](../Media/EgoLanes_GIF.gif) 

## [model_library](./model_library/)
To learn more about our open-source models and to download the free pre-trained model weights, please go to the model_library, where you will find various open-source models for different tasks, ranging from object detection to semantic segmentation. We also share benchmark results of the models on open-source datasets.

## [visualizations](./visualizations/)
If you want to try out one of the models, then please visit the visualizations folder, where you will find examples of how to run model inference on a single image as well as a video

## [model_components](./model_components/)
To understand the architectural layers which comprise each of our models, please enter the model_components folder, in which there are definitions of the various layers and modular blocks which we have used to construct our custom models

## [training](./training/)
In the training folder, we share our open-source model training pipelines which are configured to work with open-source data. You retrain our models on open-source data or modify the training pipelines to use your custom data. Each model has a "trainer" class which contains all of the main training implementation such as model loading, loss function definition, applying augmentations, loss calculation, and logging to TensorBoard. Each model also has a "train" script which calls functions from the "trainer" class and is responsible for orchestrating the training using various datasets.

## [data_parsing](./data_parsing/)
We utilize open-source datasets to train our models. Since each open-source dataset may have a different ground truth format, we provide various dataset parsing scripts which are used to prepare our custom ground truth datasets specific for the learning task of each model.

## [data_utils](./data_utils/)
The data_utils folder provides data loaders for each model which help parse and load the specific open-source datasets used to train that model. This folder also contains an augmentations class which applies various image noise transforms to make models more robust and generalizable during training. It also provides other simple data helper classes.

## [exports](./exports/)
Within the exports folder, you can find more information about the 'Quanty' open-source model quantization framework which can compress large model files into smaller models while preserving model inference accuracy. This folder also contains simple utilities for converting model formats.

## [inference](./inference/)
In the inference folder, there are helper classes which are used to perform inference with pre-trained models.





