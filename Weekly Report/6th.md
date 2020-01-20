# Dataset Preparation

## Shoe_V2 Description

- including 6648 sketches and 2000 images in total. Each photo has 3 or more corresponing sketches. 
- Training/testing splits (photo/sketch): 
	- Training: 1800/5982
	- Testing: 200/666
- All photos are from the online shopping website: https://www.office.co.uk/. Sketches are collected via Amazon Mechanical Turk.
- All sketches are provided in SVG format. You can convert them to PNG using Inkscape.
- Each photo image is named by its productID: for example, '2429245009.png', '2429245009' is the productID of the shoe in the image. While for each sketch image, '2429245009_1.svg', '1' is an instanceID we recorded during collection. 

If one wants to use this dataset, please cite the project page: http://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html



## Conversion & Split up

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

