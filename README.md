# fish_auto_landmark
This repository is for training ML Unet to auto landmark and segment fish images in 2D.
- full resolutioni images are in the images folder
- reduced images are in the reduced folder
- landmakrs are in the fcsv folder
- labelmaps for segmentations are in the labelmaps folder


to view training progress:

```
plot( ts( na.omit( read.csv("models/fish_seg.csv" ) )[,2]))
```

