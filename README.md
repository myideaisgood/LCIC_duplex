Implementation of "Lossless Color Image Compresion by Joint Prediction of Pixel and Coding Context Using Duplex Neural Networks"

Hochang Rhee, Yeong Il Jang, Seyun Kim, and Nam Ik Cho

## Environments
- Ubuntu 18.04
- [Tensorflow 1.13.1](http://www.tensorflow.org/)
- CUDA 10.0.130 & cuDNN 7.6.5
- Python 3.7.7

You can type the following command to easily build the environment.
Download 'lcic_duplex.yml' and type the following command.

```
conda env create -f lcic_duplex.yml
```

## Abstract

This paper presents a new lossless color image compression method based on the learning of pixel values and coding contexts through a multiplayer perceptron (MLP). The pixel prediction error and contexts are forwarded to an adaptive arithmetic encoder, like the conventional lossless compression schemes. The neural net-based prediction has long been attempted for the lossless compression, and recently convolutional neural networks (CNNs) are also adopted for the lossy/lossless coding. While the existing MLP-based lossless compression schemes focused only on the accurate pixel prediction, we jointly predict the pixel values and coding contexts. We also propose channel-wise progressive learning, residual learning, and duplex network in this MLPbased framework, which leads to improved coding gain compared to the conventional method. Experiments show that the proposed method performs better or comparable to the recent learningbased and conventional compression methods.

## Related Work
[LCIC (TIP 2013)] Hiearchical Predication and Context Adaptive Coding for Lossless Color Image Compression <a href="https://github.com/jyicu/LCIC">LCIC</a>

[FLIF (ICIP 2016)] Free Lossless Image Format Based on MANIAC Compression <a href="https://github.com/FLIF-hub/FLIF">FLIF</a>

[L3C (CVPR 2019)] Practical Full Resolution Learned Lossless Image Compression <a href="https://github.com/fab-jul/L3C-PyTorch">L3C</a>

[CWPLIC (ICIP 2020)] Channel-wise Progressive Learning for Lossless Image Compression <a href="https://github.com/myideaisgood/CWPLIC">CWPLIC</a>

## Proposed Method

### <u>Overall framework of proposed method</u>

<p align="center"><img src="figure/method1.png" width="700"></p>

We first apply a reversible color transform proposed in to the input RGB images to decorrelate the color components. Then, for each encoding pixel, prediction for the pixel value and coding context are simultaneously generated in the raster scan order. Afterward,
to utilize the AAC as our entropy coder, we quantize the obtained real-valued coding contexts into N steps, where the level of the quantized coding context is proportional to the magnitude of the local activity. In AAC, individual entropy coder is employed for each quantized coding context, because the statistics of prediction error differs depending on the local activity. Finally, the prediction error is compressed into a bitstream based on the corresponding quantized coding context through the AAC.

<p align="center"><img src="figure/method2.png" width="700"></p>

The pixel estimator generally shows a good estimation performance in smooth areas and relatively poor in texture regions. In other words, the network shows different behavior depending on the area, and thus it would be beneficial to have a different network for each type of region. In this respect, we propose a duplex network, where one is specialized for the smooth and the other for the textured.

## Dataset
Train Dataset

[DIV2K] (https://data.vision.ee.ethz.ch/cvl/DIV2K/)

Test Dataset

[DIV2K] (https://data.vision.ee.ethz.ch/cvl/DIV2K/)

[FLICKR2K] dataset/FLICKR2K 100 images randomly sampled from (https://github.com/limbee/NTIRE2017)

[Classic] dataset/classic

## Experimental Results

**Results on compression performance**

<p align="center"><img src="figure/result_bpp.png" width="1000"></p>

Comparison of our method with other engineered and learning based codecs. We measure the performances in bits per pixel (bpp). The difference in percentage to our method is highlighted in green if our method outperforms and in red otherwise.

**Results on CPU time**

<p align="center"><img src="figure/result_time.png" width="350"></p>

Comparision of computation time (CPU time in seconds). We compared times for 512 x 512 image.

**Ablation Study**

<p align="center"><img src="figure/result_ablation.png" width="500"></p>

Ablation study of our method on DIV2K dataset.

**Prediction Error**

<p align="center"><img src="figure/result_prederror.png" width="600"></p>

Visualization of the magnitude of prediction errors compared to different methods.

**Coding Context**

<p align="center"><img src="figure/result_ctx.png" width="1000"></p>

Analysis of the predicted coding contexts.


## Brief explanation of contents

```
├── c_compression : c++ code for compressing images with MLP weights obtained from python code
    └──> LCIC_duplex   : weights & images should be saved here
├── dataset : evaluation dataset
├── executable : executable of our method
├── python_weights_training : python code for training MLP weights
    ├──> ckpt    : trained models will be saved here
    ├──> board   : tensorboard logs will be saved here
    └──> dataset/train : train data should be saved here (no test data required for MLP training)
└── lcic_duplex.yml : enviroment of our method

```

## Guidelines for Codes

### Train (Weight Training)
Code in directory "python_weights_training"

Run the following command for training the network.

```
python main.py
```

Adjust the argument "crop_size" to handle train images that are of small size.

```
python main.py --crop_size=128
```

MLP weights of channel Y,U,V for smooth/texture network will be saved in **weights_smooth_y.txt**, **weights_smooth_u.txt**, **weights_smooth_v.txt, weights_texture_y.txt**, **weights_texture_u.txt**, **weights_texture_v.txt**.

### Test (Compression)
Code in directory "c_compression".

Place the MLP weights **weights_smooth_y.txt**, **weights_smooth_u.txt**, **weights_smooth_v.txt, weights_texture_y.txt**, **weights_texture_u.txt**, **weights_texture_v.txt** in directory under "c_compression/LCIC_duplex/.

Place the desired evaluation data in "c_compression/LCIC_duplex/data/".
The evaluation data needs to be in the form of ppm.

Run the code.

### Test (Excutable)
File in directory "executable".

**Encoding**
```
LCIC_duplex.exe e [source_file (ppm)] [compressed_file (bin)]
```

Creates compressed_file_y.bin, compressed_file_u.bin, compressed_file_v.bin.

**Decoding**
```
LCIC_duplex.exe d [compressed_file (bin)] [decoded_file (ppm)]
```

In case of compreesed_file_y.bin, compressed_file_u.bin, compressed_file_v.bin, the input should be compressed_file.bin.
Creates decoded_file.ppm.
