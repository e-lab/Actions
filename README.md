# Gesture Recognition from Videos

This repository contains scripts to train deep neural network for gesture recognition using video dataset.
Our aim is to use the temporal coherency present in video data and stabilize neural network's output probabilities.

### Input data structure

```
data/
├── class_1
│   ├── video_1.mp4
|   ...
|   └── video_m.mp4
└── class_2
    ├── video_1.mp4
    ...
    └── video_n.mp4
```

**Step 1: Prepare dataset**

> Folder containing videos (following the above structure) will be used as an input for `prepData.py` script.
This script will convert these videos into tensors with `.pyt` extensions.

**Step 2: Train neural network**

> Use `main.py` script to train your neural network. Make sure the input image resolution is the same as that provided in the above step.
Also, dataset location, place to save model, batch size are some of the inputs which must be provided to the training script.

```
python main.py --data /data/in/tensor --save /save/trained/model
```
