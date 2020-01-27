from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, BatchNormalization
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.utils.vis_utils import plot_model

import numpy as np
import glob
import os
import datetime
import matplotlib.pyplot as plt
from PIL import Image
from random import randint, shuffle

class CycleGAN():
    def __init__(self):
        
        # input image shape
        self.image_rows = 128
        self.image_cols = 128
        self.channels = 3
        self.image_shape = (self.image_rows, self.image_cols, self.channels)
        
        # load data
        self.loadsize = 400
        self.imagesize = self.image_rows 
        self.dpath = 'data/ShoeV2/'
        
        # hyper parameter
        self.lr_D = 0.0002
        self.lr_G = 0.0002
        self.beta_1 = 0.5
        self.batch_size = 1
        self.epochs = 10
        self.save_interval = 10
        
         # Calculate output shape of D (PatchGAN)
        patch = int(self.image_rows / 2**3)
        self.patch_shape = (patch, patch, 1)
        
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
        self.D_y.compile(loss='mse', optimizer=Adam(lr=self.lr_D, beta_1=self.beta_1), loss_weights=[0.5])
        self.D_x.compile(loss='mse', optimizer=Adam(lr=self.lr_D, beta_1=self.beta_1), loss_weights=[0.5])

        # compile the composite model to train generators to fool discriminators
        self.Composite = self.composite_model(name='Composite')
        self.Composite.compile(loss=['mse','mse','mae','mae'], optimizer=Adam(lr=self.lr_G, beta_1=self.beta_1), loss_weights=[1, 1, self.lambda_forward_cycle, self.lambda_backward_cycle])
        
        

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
        
#===============================================================================
# Training

    def train(self):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((self.batch_size,) + self.patch_shape)
        fake = np.zeros((self.batch_size,) + self.patch_shape)
        
        for epoch in range(self.epochs):
            for batch_i, (imgs_X, imgs_Y) in enumerate(self.load_data()):
                
                # ----------------------
                #  Train Discriminators
                # ----------------------
                
                fake_Y = self.G.predict(imgs_X)
                fake_X = self.F.predict(imgs_Y)
                
                dY_loss_real = self.D_y.train_on_batch(imgs_Y, valid)
                dY_loss_fake = self.D_y.train_on_batch(fake_Y, fake)
                dY_loss = np.add(dY_loss_real, dY_loss_fake) 
                
                dX_loss_real = self.D_x.train_on_batch(imgs_X, valid)
                dX_loss_fake = self.D_x.train_on_batch(fake_X, fake)
                dX_loss = np.add(dX_loss_real, dX_loss_fake) 
                
                # Total disciminator loss
                d_loss = 0.5 * np.add(dY_loss, dX_loss)
                
            
                # ------------------
                #  Train Generators
                # ------------------
                
                g_loss = self.Composite.train_on_batch([imgs_X, imgs_Y],
                                                             [valid, valid, imgs_X, imgs_Y])
                
                
                elapsed_time = datetime.datetime.now() - start_time
                
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %05f, adversarial: %05f, reconstr: %05f] time: %s " \
                                                                        % ( epoch, self.epochs,
                                                                            batch_i, self.n_batches,
                                                                            d_loss,
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            elapsed_time))
                # If at save interval => save generated image samples
                if batch_i % self.save_interval == 0:
                    self.sample_images(epoch, batch_i)
    
#===============================================================================
# Data Loader
    
    def read_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.loadsize, self.loadsize), Image.BICUBIC)
        img = np.array(img)
        assert img.shape == (self.loadsize, self.loadsize, 3)
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        # random jitter
        w_offset = h_offset = randint(0, max(0, self.loadsize - self.imagesize - 1))
        img = img[h_offset:h_offset + self.imagesize, w_offset:w_offset + self.imagesize, :]
        # horizontal flip
        if randint(0, 1):
            img = img[:, ::-1]
        return img
        
    def load_data(self):
        # configure traning dataset path
        train_A = glob.glob(self.dpath+'trainA/*')
        train_B = glob.glob(self.dpath+'trainB/*')
      
        self.n_batches = int(min(len(train_A), len(train_B)) / self.batch_size)
        total_samples = self.n_batches * self.batch_size
        
        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        train_A = np.random.choice(train_A, total_samples, replace=False)
        train_B = np.random.choice(train_B, total_samples, replace=False)
        
        for i in range(self.n_batches-1):
            batch_A = train_A[i*self.batch_size:(i+1)*self.batch_size]
            batch_B = train_B[i*self.batch_size:(i+1)*self.batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.read_image(img_A)
                img_B = self.read_image(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
            
            yield np.array(imgs_A), np.array(imgs_B)

#===============================================================================
# Save samples 
    
    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dpath, exist_ok = True)
        
        # configure testing dataset path
        val_A = glob.glob(self.dpath+'testA/*')
        val_B = glob.glob(self.dpath+'testB/*')
        
        val_A = np.random.choice(val_A, size=self.batch_size)
        val_B = np.random.choice(val_B, size=self.batch_size)
        
        imgs_A, imgs_B = [], []
        for i in range(self.batch_size):
            path_A = val_A[i*self.batch_size:(i+1)*self.batch_size]
            path_B = val_B[i*self.batch_size:(i+1)*self.batch_size]
            for img_A, img_B in zip(path_A, path_B):
                img_A = self.read_image(img_A)
                img_B = self.read_image(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
        
        imgs_A = np.array(imgs_A)
        imgs_B = np.array(imgs_B)

        # Translate images to the other domain
        fake_B = self.G.predict(imgs_A)
        fake_A = self.F.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.F.predict(fake_B)
        reconstr_B = self.G.predict(fake_A)
        
        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
        
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        
        titles = ['Original', 'Translated', 'Reconstructed']
        r, c = 2, 3
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dpath, epoch, batch_i))
        plt.close()
        
if __name__ == '__main__':
	# # create the model
	GAN = CycleGAN()
	GAN.train()
	# model = GAN.Composite
	# # summarize the model
	# model.summary()
	# # plot the model
	# plot_model(model, to_file='composite_model_plot.png', show_shapes=True, show_layer_names=True)