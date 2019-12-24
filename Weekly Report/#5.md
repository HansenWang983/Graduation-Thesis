# 实验思路

## 基本流程 

每个阶段按照两周时间准备，时间允许则进行一次迭代，剩余的时间重构代码。

#### 阶段一

1. 将数据集划分为两个 domain，一类是 sketch，一类是  photo，利用 CycleGAN 进行无监督学习



#### 阶段二

1. 尝试不同于 CycleGAN 中基于 ResNet 的生成器结构，如 U-net
2. 或者将图像转换任务分阶段进行，第一阶段为 shape 的细节补充，采用之前的 CycleGAN 方式；第二阶段为 colorization，可以借鉴 pix2pix 等一些图像翻译网络直接进行 encode & decode，在其中还可以添加一张 reference photo 用于颜色的参考，通过 AdaIN 的方法在保留 content strcture feature 的同时加入 color feature



#### 阶段三

1. 添加不同的 loss function （重建损失，特征损失，style 损失，total variation 损失等）做 ablation study
2. 使用数据增强的方法增加模型鲁棒性，即针对复杂和简单的手绘图都有较好的生成效果，或者增加其他的数据集资源。如果失败，则尝试使用注意力机制使得对于不同风格的输入，只关注于重要的部分，这个可以在生成器的 downsample layer 之后加入一些卷积层得到 attention mask （待研究）



#### 阶段四

1. 总结一些常用的 metrics 并进行 evaluation
2. 寻找一些 baselines 进行定量和定性的比较



## 数据集

- [x] [ShoeV2](https://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html)

- [ ] [QuickDraw](https://github.com/googlecreativelab/quickdraw-dataset)
- [ ] [Sketchy](http://sketchy.eye.gatech.edu/)
- [ ] [TU-Berlin](http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/)



