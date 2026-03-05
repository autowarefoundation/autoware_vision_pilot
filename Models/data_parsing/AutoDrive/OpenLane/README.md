# OpenLane-V1

OpenLane is scaled 3D lane dataset. Dataset collects contents from public perception dataset, providing lane and closest-in-path object(CIPO) annotations for 1000 segments. OpenLane owns 200K frames and over 880K carefully annotated lanes. In v1.0, OpenLane contains the annotations on Waymo Open Dataset.

For the purpose of `Closest In-Path Object` detection OpenLane dataset is preprocessed and directory structure is converted into the following format

```
dataset
├── images
│   ├── train
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── …
│   ├── val
│   │   ├── imgA.jpg
│   │   └── …
│   └── test
│       ├── imgX.jpg
│       └── …
└── labels
    ├── train
    │   ├── img1.txt
    │   ├── img2.txt
    │   └── …
    ├── val
    │   ├── imgA.txt
    │   └── …
    └── test
        ├── imgX.txt
        └── …
```
And labels are converted into format: 

```
<class_id> <x_center> <y_center> <width> <height>
```

Usage:

```
python3 converter.py -d <DATASET_DIR>
```

Test converted dataset:

```
python3 test_conversion.py -i <IMAGE_PATH>
```