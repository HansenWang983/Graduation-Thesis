
# coding: utf-8

# In[21]:


from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
import glob
import os


# In[22]:


svgs = glob.glob('../ShoeV2_F/ShoeV2_sketch/*.svg') 
# print(len(svgs))


# In[23]:


pngpath = 'data/ShoeV2/'
if not os.path.exists(pngpath):
    os.makedirs(pngpath)

for s in svgs:
    portion = os.path.splitext(s)
    png = pngpath + portion[0][26:] + '.png'
    drawing = svg2rlg(s)
    renderPM.drawToFile(drawing, png, fmt='PNG')
    


# In[59]:


trainA_path = os.path.join('data/ShoeV2', 'trainA/')
trainB_path = os.path.join('data/ShoeV2', 'trainB/')
testA_path = os.path.join('data/ShoeV2',  'testA/')
testB_path = os.path.join('data/ShoeV2',  'testB/')


# In[62]:


trainpath = os.path.join('../ShoeV2_F', 'ShoeV2_photo/')
testpath = os.path.join('data', 'ShoeV2/')

with open('../ShoeV2_F/sketch_train.txt') as f:
    mylist = f.read().splitlines() 
    for line in mylist:
        portion = os.path.splitext(line)
        name = portion[0] + '.png'
        os.replace('data/ShoeV2/' + name, trainB_path + name )

