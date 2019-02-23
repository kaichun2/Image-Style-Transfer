# CS230 project

An implementation of style transfer in TensorFlow.

We choose to use SqueezeNet for efficiency, since the pre-trained model is very small (~5MB full, and ~3MB without classifier). It offers decrease in optimization iteration time and decrease in GPU memory consumption.

The SqueezeNet is described in squeezenet.py, with weights in sqz_full.mat. 

nst_utils.py describes some utility functions we need in transfer, and is adopted from CS230.
