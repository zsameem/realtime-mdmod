# Minimum Delay Moving Object Detection in Realtime

Our code is released only for scientific or personal use.
Please contact us for commercial use.

Compiling
=========
## Opencv
This software relies heavily on opencv. It was written and tested on opencv version 2.4.11. Install opencv frome [here](https://github.com/opencv/opencv).

## Flownet-2
To generate optical flow, flownet2 is required. It is included within this package. To compile the flownet2 simply run following commands.

```
$ cd ./flownet2_caffe
$ make  -j 5 all tools pycaffe 
```
To download the pre trained model for flownet run the following command.

```
$ cd flownet2_caffe/models
$ bash download-models.sh
```

Use the following command to set the environment.
```
$ source flownet2_caffe/set-env.sh
```
Flownet 2 is based on caffe. If you are having trouble compiling it, refer to [flownet2 documentation](https://github.com/lmb-freiburg/flownet2) and [caffe dicumentation](https://github.com/BVLC/caffe).

## Running
To run the program for realtime quickest moving change detection execute the following command.
