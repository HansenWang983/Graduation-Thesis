
# coding: utf-8

# In[9]:


from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, BatchNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model


class CycleGAN():
    def __init__(self):
        # input image shape
        self.image_rows = 128
        self.image_cols = 128
        self.channels = 1
        self.image_shape = (self.image_rows, self.image_cols, self.channels)
        
        self.use_patchgan = True
        self.n_resnet = 6
        
        # Hyper parameters
        # Cycle-consistency loss weights
        self.lambda_forward_cycle = 10.0
        self.lambda_backward_cycle = 10.0
        
        # build the two generators and two discriminators used in the CycleGAN
        # Domain X -> Y
        self.G = self.generator(name='G')
        # Domain Y -> X
        self.F = self.generator(name='F')
        
        # predict Y
        self.D_y = self.discriminator(name='D_y')
        # predict X
        self.D_x = self.discriminator(name='D_x')
        
        # compile the discriminators to train discriminators 
        # In practice, we divide the objective by 2 while optimizing D, which slows down the rate at which D learns, relative to the rate of G
        self.D_y.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
        self.D_x.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

        # compile the composite model to train generators to fool discriminators
        self.Composite = self.composite_model(name='Composite')
        self.Composite.compile(loss=['mse','mse','mae','mae'], optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[1, 1, self.lambda_forward_cycle, self.lambda_backward_cycle])
        
        

#===============================================================================
# Architecture functions

    # Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k ﬁlters and stride 2
    def ck(self, layer_input, k, use_normalization, stride):
        x = Conv2D(filters=k, kernel_size=4, strides=stride, padding='same')(layer_input)
        if use_normalization:
        # The “axis” argument is set to -1 to ensure that features are normalized per feature map
            x = InstanceNormalization(axis=-1)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x
    
    # Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of ﬁlters on both layer
    def Rk(self, layer_input, k):
        # 1st layer
        # Same padding is used instead of reflection padded recommended in the paper for simplicity
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(layer_input)
        x = InstanceNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        
        # 2nd layer
        x = Conv2D(filters=k, kernel_size=3, strides=1, padding='same')(x)
        x = InstanceNormalization(axis=-1)(x)
        
        # concatenate merge channel-wise with input layer
        x = Concatenate()([x, layer_input])
        return x
    
    # c7s1-k denote a 7×7 Convolution-InstanceNormReLU layer with k ﬁlters and stride 1
    def c7Ak(self, layer_input, k):
        x = Conv2D(filters=k, kernel_size=7, strides=1, padding='same')(layer_input)
        x = InstanceNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        return x
    
    # dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k ﬁlters and stride 2
    def dk(self, layer_input, k):
        x = Conv2D(filters=k, kernel_size=3, strides=2, padding='same')(layer_input)
        x = InstanceNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        return x
    
    # uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k ﬁlters and stride 1/2
    def uk(self, layer_input, k):
        # this matches fractinoally stided with stride 1/2
        x = Conv2DTranspose(filters=k, kernel_size=3, strides=2, padding='same')(layer_input)
        x = InstanceNormalization(axis=-1)(x)
        x = Activation('relu')(x)
        return x
     
#===============================================================================
# Models

    # define the 70x70 patchgan discriminator model
    def discriminator(self, name = None):
        # Specify input
        input_image = Input(shape=self.image_shape)

        # Layer 1 (#Instance normalization is not used for this layer)
        x = self.ck(input_image, 64, False, 2)
        # Layer 2
        x = self.ck(x, 128, True, 2)
        # Layer 3
        x = self.ck(x, 256, True, 2)
        # Layer 4
        x = self.ck(x, 512, True, 1)
        
        # Output Layer
        if self.use_patchgan:
            x = Conv2D(filters=1, kernel_size=4, strides=1, padding='same')(x)
        else:
            x = Flatten()(x)
            x = Dense(1)(x)
            
        model = Model(inputs = input_image, outputs = x, name = name)
        return model
    
    # 6-resnet block version
    def generator(self, name = None):
        # Specify input
        input_image = Input(shape=self.image_shape)
        
        # Layer 1 
        x = self.c7Ak(input_image, 64)
        # Layer 2
        x = self.dk(x, 128)
        # Layer 3
        x = self.dk(x, 256)
        # Layer 4-9
        for _ in range(self.n_resnet):
            x = self.Rk(x, 256)
        
        # Layer 10
        x = self.uk(x, 128)
        # Layer 11
        x = self.uk(x, 64)
            
        # Layer 12, c7s1-1
        x = Conv2D(self.channels, kernel_size=7, strides=1, padding='same')(x)
        x = InstanceNormalization(axis=-1)(x)
        # pixel values are in the range [-1, 1]
        output_image = Activation('tanh')(x) 
        
        model = Model(inputs = input_image, outputs = output_image, name = name)
        return model
    
    # For the composite model we will only train the generators
    def composite_model(self, name = None):
       
        # ensure generators we're updating is trainable
        self.G.trainable = True
        self.F.trainable = True
        # mark discriminator as not trainable
        self.D_y.trainable = False
        self.D_x.trainable = False
        
        # Input images from both domains
        img_X = Input(shape=self.image_shape)
        img_Y = Input(shape=self.image_shape)
        
        # Translate images to the other domain
        fake_Y = self.G(img_X)
        fake_X = self.F(img_Y)
        
        # Translate images back to original domain
        reconstr_X = self.F(fake_Y)
        reconstr_Y = self.G(fake_X)
        
        # Discriminators determines validity of translated images to compute Adversarial Loss
        valid_Y = self.D_y(fake_Y)
        valid_X = self.D_x(fake_X)
        
        model = Model(inputs=[img_X, img_Y], outputs=[valid_Y, valid_X, reconstr_X, reconstr_Y])
        
        return model
        
        
        


# In[13]:


# # create the model
# GAN = CycleGAN()
# model = GAN.Composite
# # summarize the model
# model.summary()
# # plot the model
# plot_model(model, to_file='composite_model_plot.png', show_shapes=True, show_layer_names=True)

