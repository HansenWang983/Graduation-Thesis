## 周报汇总

### 第三周

- 基于条件 GAN 的手绘图到卡通图片生成：
  - 生成器使用 Unet 结构，相比简单的 encoder-decoder 网络在卷积时会损失较多的信息，Unet 可以通过 concatenate 即保存了一定的 sketch 信息，也保存训练得到的 color 信息。
  - 除了对抗损失外，增加更多种类的 loss，比如 feature loss 和  total variation loss（保证前景颜色和背景颜色不会发生较多的波动）
- SPADE：
  - 提出了一种新的规范化方法用于 segmentation mask 到 photo 的生成，保证在输入 mask 的 label 较少的情况下不会因 BN 造成信息丢失，仍然可以产生和 label 区域一样的自然图片。



### 第四周

- 无监督的风格迁移和手绘图到自然图片生成的两篇论文：
  - 经典的 CycleGAN 可以在不需要一对一图片的情况下进行图片风格的转换，只需要分为两个数据集，一个是 source，一个是 target，通过训练两个生成器和判别器完成双向 mapping，较于 pix2pix 等图像翻译有着更大的自由度。但同时提出了 cycle consistency loss，保证原图像的内容不会发生太大的改变。
  - 由于 sketch 具有很大的抽象性，但是 photo 包含了很多细节，所以可以将 sketch-to-photo 的问题转换为 style transfer 的问题，并且通过分阶段生成保证轮廓细节和颜色都较为准确地表达和一定的自由度呈现，而不陷入 mode collapse，即对于不同输入，会产生相同颜色的输出。



### 第五周

- 尝试完成一类 sketch 转换的实验：
  - [下载 ShoeV2 dataset](https://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html)
  - 构建并运行 Keras 版本的 CycleGAN 



### 第六周

- 数据预处理：
  - ShoeV2 数据集 svg 转 png
  - 划分训练集和测试集
  - 调整图片大小
  - 数据导入



### 第七周

- CycleGAN Keras 实现第一部分：
  - 生成器
  - 判别器
  - 组合模型



### 第八周

- CycleGAN Keras 实现第二部分：
  - 训练
  - 测试





