# Conditional Batch Normalization

## CBN 概述

传统的 Batch Normalization (BN) 公式为：

![[公式]](https://www.zhihu.com/equation?tex=y+%3D+%5Cfrac%7Bx-%5Cmathbb%7BE%7D%5Bx%5D%7D%7B%5Csqrt%7B%5Cmathrm%7BVar%7D%5Bx%5D%2B%5Cepsilon%7D%7D%5Ccdot+%5Cgamma%2B%5Cbeta%5Ctag%7B1%7D)

其中的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta+) 都是网络层的参数，需要通过损失函数反向传播来学习。Conditional Batch Normalization (CBN)中，输入的 feature 也要先减均值，再除标准差；但是做线性映射时，乘以的缩放因子变为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma_%7Bpred%7D) ，加的偏置变为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_%7Bpred%7D+) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma_%7Bpred%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_%7Bpred%7D) 是把 feature 输入一个小神经网络（多层感知机），前向传播得到的网络输出，而不是学习得到的网络参数（网络参数独立于输入 feature，而 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma_%7Bpred%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_%7Bpred%7D) 取决于输入的 feature）。由于 ![[公式]](https://www.zhihu.com/equation?tex=%5Cgamma_%7Bpred%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_%7Bpred%7D) 依赖于输入的 feature 这个 condition，因此这个改进版的 Batch Normalization 叫做 Conditional Batch Normalization。



## BN 解决的问题

1. 网络中每层输入数据的分布相对稳定，加速模型学习速度
2. 模型对网络中的初始化权重和训练时的学习率不那么敏感，简化调参过程，使得网络学习更加稳定
3. 饱和性激活函数（例如sigmoid，tanh等）的输入落在梯度非饱和区，缓解梯度消失问题
4. 一定的正则化效果，减少训练出的模型过拟合



