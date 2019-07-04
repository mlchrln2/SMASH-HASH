Smash-Hash: Image Recommendation Network
================================================
Image Recommendation Network that captions images and then recommends similar images with similar captions to the user. This project is still in development and needs minor improvements to the captioning network and reformulation of the Graphical Convolution Network.

Dependencies:
------------
* torch
* torchvision
* tensorboardX
* nltk
* h5py
* PIL
* gc

ImageNet: Image Captioning Network
================================================

Data Preparation (for development):
------------
```bash
# To create data files for faster training and validation:
python3 DataWriter.py
```

Parameter Setup (for development):
------------
``` bash
# Edit this file to toggle hyperparameters:
python3 HyperParameters.py
```

Training Setup (for development):
------------
```bash
# To train a new model:
python3 ImageTrain.py

# To continue training an old model:
python3 ImageTrain.py /path/to/file
```

Model Validation:
------------
```bash
# To judge the effectiveness of the model:
python3 ImageVal.py /path/to/file greedy
```
