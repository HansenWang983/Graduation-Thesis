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

c7s1-64,d128,d256,R256,R256,R256, R256,R256,R256,u128,u64,c7s1-3



## Discriminator

The discriminator model is responsible for taking a real or generated image as input and predicting whether it is real or fake.

The discriminator model is implemented as a 70 × 70 PatchGANs model, which aim to classify whether 70 × 70 overlapping image patches are real or fake. Instead of outputting a single value like a traditional discriminator model, the PatchGAN discriminator model can output a square or one-channel feature map of predictions. The 70×70 refers to the effective receptive field of the model on the input, not the actual shape of the output feature map. The receptive field of a convolutional layer refers to the number of pixels that one output of the layer maps to in the input to the layer. 

Accordingly, Let

- Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k ﬁlters and stride 2.

The architecture for the discriminator is as follows:

C64-C128-C256-C512

Specificlly, using leaky ReLUs with a slope of 0.2 for the first C64 layer, instead of using InstanceNorm.

For the 128×128 images were used as input, then the size of the output feature map of activations would be 8×8.



## Least Squares and Cycle Loss

