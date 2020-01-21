# CycleGAN Keras Implementation

## CycleGAN Architechture

![arch](Assets/7/arch.jpg)

- 2 translator, G: X -> Y, F: Y -> X
- 2 corresponding discriminator, Dy for predicting Y and Y^, Dx for predicting X and X^

------



## Generator

The CycleGAN Generator model takes an image as input and generates a translated image as output.

The model uses a sequence of downsampling convolutional blocks to encode the input image, a number of residual network ([ResNet](https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/)) convolutional blocks to transform the image, and a number of upsampling convolutional blocks to generate the output image.

Following the naming convention used in the [Johnson et al.’s Github repository,](https://github.com/jcjohnson/fast-neural-style) Let

- c7s1-k denote a 7×7 Convolution-InstanceNormReLU layer with k ﬁlters and stride 1
- dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k ﬁlters and stride 2
- Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of ﬁlters on both layer
- uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k ﬁlters and stride 1/2

For the 128*128 input image, the architechture (consists of 6 residual blocks) is as follows:

c7s1-64,d128,d256,R256,R256,R256, R256,R256,R256,u128,u64,c7s1-1

Summary of the generator model

```python
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_3 (InputLayer)            (None, 128, 128, 1)  0                                            
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 128, 128, 64) 3200        input_3[0][0]                    
__________________________________________________________________________________________________
instance_normalization_4 (Insta (None, 128, 128, 64) 128         conv2d_6[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 128, 128, 64) 0           instance_normalization_4[0][0]   
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 64, 64, 128)  73856       activation_1[0][0]               
__________________________________________________________________________________________________
instance_normalization_5 (Insta (None, 64, 64, 128)  256         conv2d_7[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 64, 64, 128)  0           instance_normalization_5[0][0]   
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 256)  295168      activation_2[0][0]               
__________________________________________________________________________________________________
instance_normalization_6 (Insta (None, 32, 32, 256)  512         conv2d_8[0][0]                   
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 32, 256)  0           instance_normalization_6[0][0]   
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 32, 32, 256)  590080      activation_3[0][0]               
__________________________________________________________________________________________________
instance_normalization_8 (Insta (None, 32, 32, 256)  512         conv2d_10[0][0]                  
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 32, 32, 512)  0           instance_normalization_8[0][0]   
                                                                 activation_3[0][0]               
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 32, 32, 256)  1179904     concatenate_1[0][0]              
__________________________________________________________________________________________________
instance_normalization_10 (Inst (None, 32, 32, 256)  512         conv2d_12[0][0]                  
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, 32, 32, 768)  0           instance_normalization_10[0][0]  
                                                                 concatenate_1[0][0]              
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 32, 32, 256)  1769728     concatenate_2[0][0]              
__________________________________________________________________________________________________
instance_normalization_12 (Inst (None, 32, 32, 256)  512         conv2d_14[0][0]                  
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 32, 32, 1024) 0           instance_normalization_12[0][0]  
                                                                 concatenate_2[0][0]              
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 32, 32, 256)  2359552     concatenate_3[0][0]              
__________________________________________________________________________________________________
instance_normalization_14 (Inst (None, 32, 32, 256)  512         conv2d_16[0][0]                  
__________________________________________________________________________________________________
concatenate_4 (Concatenate)     (None, 32, 32, 1280) 0           instance_normalization_14[0][0]  
                                                                 concatenate_3[0][0]              
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 32, 32, 256)  2949376     concatenate_4[0][0]              
__________________________________________________________________________________________________
instance_normalization_16 (Inst (None, 32, 32, 256)  512         conv2d_18[0][0]                  
__________________________________________________________________________________________________
concatenate_5 (Concatenate)     (None, 32, 32, 1536) 0           instance_normalization_16[0][0]  
                                                                 concatenate_4[0][0]              
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 32, 32, 256)  3539200     concatenate_5[0][0]              
__________________________________________________________________________________________________
instance_normalization_18 (Inst (None, 32, 32, 256)  512         conv2d_20[0][0]                  
__________________________________________________________________________________________________
concatenate_6 (Concatenate)     (None, 32, 32, 1792) 0           instance_normalization_18[0][0]  
                                                                 concatenate_5[0][0]              
__________________________________________________________________________________________________
conv2d_transpose_1 (Conv2DTrans (None, 64, 64, 128)  2064512     concatenate_6[0][0]              
__________________________________________________________________________________________________
instance_normalization_19 (Inst (None, 64, 64, 128)  256         conv2d_transpose_1[0][0]         
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 64, 64, 128)  0           instance_normalization_19[0][0]  
__________________________________________________________________________________________________
conv2d_transpose_2 (Conv2DTrans (None, 128, 128, 64) 73792       activation_10[0][0]              
__________________________________________________________________________________________________
instance_normalization_20 (Inst (None, 128, 128, 64) 128         conv2d_transpose_2[0][0]         
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 128, 128, 64) 0           instance_normalization_20[0][0]  
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 128, 128, 1)  3137        activation_11[0][0]              
__________________________________________________________________________________________________
instance_normalization_21 (Inst (None, 128, 128, 1)  2           conv2d_21[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 128, 128, 1)  0           instance_normalization_21[0][0]  
==================================================================================================
Total params: 14,905,859
Trainable params: 14,905,859
Non-trainable params: 0
```

The plot of the generator model is also created, showing the skip connections in the ResNet blocks.

![generator_model_plot](Assets/7/generator_model_plot.png)

------



## Discriminator

The discriminator model is responsible for taking a real or generated image as input and predicting whether it is real or fake.

The discriminator model is implemented as a 70 × 70 PatchGANs model, which aim to classify whether 70 × 70 overlapping image patches are real or fake. Instead of outputting a single value like a traditional discriminator model, the PatchGAN discriminator model can output a square or one-channel feature map of predictions. The 70×70 refers to the effective receptive field of the model on the input, not the actual shape of the output feature map. The receptive field of a convolutional layer refers to the number of pixels that one output of the layer maps to in the input to the layer. 

Accordingly, Let

- Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k ﬁlters and stride 2.

The architecture for the discriminator is as follows:

C64-C128-C256-C512

**Specificlly, using leaky ReLUs with a slope of 0.2 for the first C64 layer, instead of using InstanceNorm. Note that the final hidden layer C512 is with a [1×1 stride](https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/), and an output layer C1, also with a 1×1 stride.**

For the 128×128 images were used as input, then the size of the output feature map of activations would be 16×16.

```python
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 128, 128, 1)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 64, 64, 64)        1088      
_________________________________________________________________
leaky_re_lu_1 (LeakyReLU)    (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 32, 128)       131200    
_________________________________________________________________
instance_normalization_1 (In (None, 32, 32, 128)       256       
_________________________________________________________________
leaky_re_lu_2 (LeakyReLU)    (None, 32, 32, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 256)       524544    
_________________________________________________________________
instance_normalization_2 (In (None, 16, 16, 256)       512       
_________________________________________________________________
leaky_re_lu_3 (LeakyReLU)    (None, 16, 16, 256)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 512)       2097664   
_________________________________________________________________
instance_normalization_3 (In (None, 16, 16, 512)       1024      
_________________________________________________________________
leaky_re_lu_4 (LeakyReLU)    (None, 16, 16, 512)       0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 16, 16, 1)         8193      
=================================================================
Total params: 2,764,481
Trainable params: 2,764,481
Non-trainable params: 0
```

![discriminator_model_plot](Assets/7/discriminator_model_plot.png)

------



## Least Squares Loss (L2)

> *we replace the negative log likelihood objective by a least-squares loss. This loss is more stable during training and generates higher quality results.*
>
> — [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593), 2017.



