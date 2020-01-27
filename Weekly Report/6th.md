# Dataset Preparation

## Shoe_V2 Description

- including 6648 sketches and 2000 images in total. Each photo has 3 or more corresponing sketches. 
- Training/testing splits (photo/sketch): 
	- Training: 1800/5982
	- Testing: 200/666
- All photos are from the online shopping website: https://www.office.co.uk/. Sketches are collected via Amazon Mechanical Turk.
- All sketches are provided in SVG format. You can convert them to PNG using Inkscape.
- Each photo image is named by its productID: for example, '2429245009.png', '2429245009' is the productID of the shoe in the image. While for each sketch image, '2429245009_1.svg', '1' is an instanceID we recorded during collection. 

If you wants to use this dataset, please cite the project page: http://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html



## Convert to PNG & Split up the training and testing set 

```python
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import glob
import os

svgs = glob.glob('../ShoeV2_F/ShoeV2_sketch/*.svg') 
# print(len(svgs))

pngpath = 'data/ShoeV2/'
if not os.path.exists(pngpath):
    os.makedirs(pngpath)

for s in svgs:
    portion = os.path.splitext(s)
    png = pngpath + portion[0][26:] + '.png'
    drawing = svg2rlg(s)
    renderPM.drawToFile(drawing, png, fmt='PNG')
    
trainA_path = os.path.join('data/ShoeV2', 'trainA/')
trainB_path = os.path.join('data/ShoeV2', 'trainB/')
testA_path = os.path.join('data/ShoeV2',  'testA/')
testB_path = os.path.join('data/ShoeV2',  'testB/')

trainpath = os.path.join('../ShoeV2_F', 'ShoeV2_photo/')
testpath = os.path.join('data', 'ShoeV2/')

with open('../ShoeV2_F/sketch_train.txt') as f:
    mylist = f.read().splitlines() 
    for line in mylist:
        portion = os.path.splitext(line)
        name = portion[0] + '.png'
        os.replace('data/ShoeV2/' + name, trainB_path + name )
```



## Resize image to 128*128 

In order to make image following the input shape of generator and discriminator, we can randomly choose height and width offset to clip the image.

```python
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
```



## Load training data 

Implement the generator of batch data that has shape of (batch_size, 128, 128, 1) 

```python
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
```



## Save sample image

Use testing dataset to get the sample from the model periodically in the training process and save the sample image. 

```python
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
```

