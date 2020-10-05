Implementation of "CHANNEL-WISE PROGRESSIVE LEARNING FOR LOSSLESS IMAGE COMPRESSION"

Hochang Rhee, Yeong Il Jang, Seyun Kim, and Nam Ik Cho

## Environments
- Ubuntu 18.04
- [Tensorflow 1.9](http://www.tensorflow.org/)
- CUDA 9.0 & cuDNN 7.6.5
- Python 3.5.5

## Abstract

This paper presents a channel-wise progressive coding system for lossless compression of color images. We follow the classical lossless compression scheme of LOCO-I and CALIC, where pixel values and coding contexts are predicted and forwarded to the entropy coder for compression. The contribution is that we jointly estimate the pixel values and coding contexts from neighboring pixels by training a simple multilayer perceptron in a residual and channel-wise progressive manner. Specifically, we obtain accurate pixel prediction along with coding contexts that reflect the magnitude of local activity very well. These results are sent to an adaptive arithmetic coder that appropriately encodes the prediction error according to the corresponding coding context. Experimental results demonstrate the effectiveness of the proposed method in high-resolution datasets.
<br><br>

## Related Work
[LCIC (TIP 2013)] Hiearchical Predication and Context Adaptive Coding for Lossless Color Image Compression <a href="https://github.com/jyicu/LCIC">LCIC</a>

[FLIF (ICIP 2016)] Free Lossless Image Format Based on MANIAC Compression <a href="https://github.com/FLIF-hub/FLIF">FLIF</a>

[L3C (CVPR 2019)] Practical Full Resolution Learned Lossless Image Compression <a href="https://github.com/fab-jul/L3C-PyTorch">L3C</a>

## Proposed Method

### <u>Overall framework of proposed method</u>

<p align="center"><img src="figure/method.png" width="700"></p>

We first apply a reversible color transform proposed in to the input RGB images to decorrelate the color components. Then, for each encoding pixel, prediction for the pixel value and coding context are simultaneously generated in the raster scan order. Afterward,
to utilize the AAC as our entropy coder, we quantize the obtained real-valued coding contexts into N steps, where the level of the quantized coding context is proportional to the magnitude of the local activity. In AAC, individual entropy coder is employed for each quantized coding context, because the statistics of prediction error differs depending on the local activity. Finally, the prediction error is compressed into a bitstream based on the corresponding quantized coding context through the AAC.

## Experimental Results

**Results on compression performance**

<p align="center"><img src="figure/result_bpp.png" width="700"></p>

Comparison of our method with other engineered and learning based codecs. We measure the performances in bits per pixel (bpp). The difference in percentage to our method is highlighted in green if our method outperforms and in red otherwise.

**Results on CPU time**

<p align="center"><img src="figure/result_time.png" width="350"></p>

Comparision of computation time (CPU time in seconds). We compared times for 512 x 512 image.

**Test Data**

[MCM]   (https://www4.comp.polyu.edu.hk/~cslzhang/CDM_Dataset.htm)

[DIV2K] (https://data.vision.ee.ethz.ch/cvl/DIV2K/)

[Open Images] (https://storage.googleapis.com/openimages/web/index.html)

## Brief explanation of contents

```
├── python_weights_training : python code for training MLP weights
    ├──> ckpt    : trained models will be saved here
    ├──> board   : tensorboard logs will be saved here
    └──> dataset : train/test data should be saved here
└── c_compression : c++ code for compressing images with MLP weights obtained from python code

```

## Guidelines for Codes

### Training

Configuration should be done in **config.py**.

[Options]
```
python main.py --gpu=[GPU_number] --epoch=[Epochs to train]
```

MLP weights of channel Y,U,V will be saved in **weights_y.txt**, **weights_u.txt**, **weights_v.txt**.

### Test (Compression)

Make sure MLP weights.txt and input images are saved at c_compression/x64/Release.

**Encoding**
```
ICIP_Compression.exe e [source file (ppm)] [compressed file (bin)]
```

**Decoding**
```
ICIP_Compression.exe d [compressed file (bin)] [decoded file (ppm)]
```

## Citation
If you use the work released here for your research, please cite this paper.

```
@inproceedings{rhee2020lcic,
  title={Channel-wise progressive learning for lossless image compression},
  author={Rhee, Hochang and Jang, Yeong Il, and Kim, Seyun and Cho, Nam Ik},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
  year={2020}
}
```
