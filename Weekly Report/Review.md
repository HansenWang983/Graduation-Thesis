#Personalized Product Search

### 个性化产品搜索的重要性

由于网上购物的日益普及以及电子商务网站上的大量产品，产品搜索为消费者提供了便利。 2015年，美国约有8％的零售额（超过3,000亿美元）来自电子商务，71％的顾客选择网上购物 (www.readycloud.com/info/ecommerce-statistics-all-retailers-should-know) 。在典型的产品搜索场景中，用户首先进行查询以获取相关产品的列表，然后浏览结果，最后选择一个或多个要购买的商品。 因此，产品搜索结果的质量直接影响了客户的满意度和交易数量。

由于购买是一种具有实际财务成本的个人行为，因此众所周知，个人偏好会直接影响客户的购买决策[29]。 先前的研究表明，产品搜索中的许多购买意图可能会受到用户的个人品味和体验的影响[35，36]。 合成数据的实验还表明，结合从产品评论和购买历史中提取的用户信息可以显着提高产品检索模型的性能[2]。 因此，直觉上个性化应该在产品搜索中具有巨大的潜力。

与传统的检索任务相比，不同顾客对同样的产品搜索结果可能有着不同的购买意愿，主观性更加强烈，所以个性化在产品搜索中的重要性就要远高于传统的检索任务（例如 Web 搜索）。一方面，用户可能仅购买了众多查询结果的一个产品，也有可能许多结果都符合用户口味，因此，仅仅完成相关性检索而不考虑用户差异将无法满足所有用户的需求。另一方面，个性化对电子商务公司有明显的好处，因为它增加了用户看到他们可能会购买的产品的机会。因此，检索相关产品不如寻找个性化的潜在商品重要。

在 Wendy W Moe. 2003. Buying, searching, or browsing: Differentiating between online shoppers using in-store navigational clickstream. Journal of consumer psychology 13, 1-2 (2003), 29–39 中，作者通过理论和实践证明了顾客在浏览电子商务网站的模式决定了他们不同的购买意愿，比如有些顾客只是为了浏览，有些仅仅是搜索，剩下的才会进行购买，因此他们会对各种营销信息做出不同的反应，从而设计出个性化的促销信息。

在 Parikshit Sondhi, Mohit Sharma, Pranam Kolari, and ChengXiang Zhai. 2018. A Taxonomy of Queries for E-commerce Search. In The 41st International ACM SIGIR (SIGIR ’18). ACM, 1245–1248. https://doi.org/10.1145/3209978.3210152 中，作者通过对产品搜索引擎日志的分析，将用户查询分为5类，并通过对不同类别的查询采用不同的检索方法来提高搜索引擎的有效性。

### 个性化产品搜索的挑战

在产品搜索中加入个性化的挑战主要是提取用户的语义信息，如产品评论，历史购买记录等。例如，关于电视机的评论可能不同于关于照相机的评论。 如果获取不到语义信息，用户评论就无法为新查询中的个性化产品搜索提供帮助，并因此可能降低产品搜索的质量。例如，在大多数有关个性化搜索算法的工作中，所有查询的结果都以相同的方式个性化。但是，对于某些查询，发出查询的每个人都在寻找相同的东西。对于其他查询，即使他们以相同的方式表达他们的需求，不同的人也希望获得截然不同的结果。在 Jaime Teevan, Susan T Dumais, and Daniel J Liebling. 2008. To personalize or not to personalize: modeling queries with variation in user intent. In Proceedings of the 31st ACM SIGIR. ACM, 163–170. 中，作者提取不同查询的特征，以及用户的反馈分辨出哪些查询可以通过加入用户的个性化信息增加相关度，哪些查询对个性化反而会有不好的效果。

再比如，当客户向产品搜索引擎提交查询“牙膏”时，他们可能想要适合其个人需求的个性化搜索结果（例如敏感牙齿）。也可能仅购买列表中最畅销的产品。所以个性化的一个主要问题是其是否总是可以提高产品搜索的质量，何时以及如何进行个性化是产品搜索的重要研究问题。

### 个性化产品搜索的相关工作

#### 向量空间模型(vector space model)

VSM 对文本内容的处理简化为向量空间中的向量运算，并且它以空间上的相似度表达语义的相似度。当文档被表示为文档空间的向量，就可以通过计算向量之间的相似性来度量文档间的相似性。文本处理中最常用的相似性度量方式是余弦距离。

VSM基本概念：

（1） 文档(Document): 泛指一般的文本或者文本中的片断(段落、句群或句子),一般指一篇文章,尽管文档可以是多媒体对象,但是以下讨论中我们只认为是文本对象,本文对文本与文档不加以区别"。

（2） 项(Term):文本的内容特征常常用它所含有的基本语言单位(字、词、词组或短语等)来表示,这些基本的语言单位被统称为文本的项,即文本可以用项集(Term List)表示为D(T1,T2,,,,Tn)其中是项,1≤k≤n"

**特征项的选择：**

项的选择必须由处理速度、精度、存储空间等方面的具体要求来决定。特征项选取有几个原则：一是应当选取包含语义信息较多，对文本的表示能力较强的语言单位作为特征项；二是文本在这些特征项上的分布应当有较为明显的统计规律性，这样将适用于信息检索、文档分类等应用系统；三是特征选取过程应该容易实现，其时间和空间复杂度都不太大。实际应用中常常采用字、词或短语作为特征项。

（3） 项的权重(TermWeight):对于含有n个项的文本D(,………)项常常被赋予一定的权重表示他们在文本D中的重要程度,即D=（，,,,······,）。这时我们说项的权重为(1≤k≤n)。

**特征值项的权重计算**：

特征项的权重计算是文本相似度计算中的一个非常重要的环节。一篇文本中的特征项数目众多，要想得到比较准确的对文本内容的数学化表示，我们需要对能显著体现文本内容特征的特征项赋予高权重，而对不能可以体现文本内容特征的特征项赋予低权重。从效率方面来说，特征项权重的计算是文本相似度计算中的主要工作，它的效率也直接影响文本相似度计算的整体效率。
经典的 TF-IDF 权重是向量空间模型中应用最多的一种权重计算方法，它以词语作为文本的特征项，每个特征项的权重由 TF 权值和 IDF 权值两个部分构成。对于文本 中的第 k 个特征项，其对应权重计算方法为：

https://blog.csdn.net/asialee_bird/article/details/81486700

（4） 向量空间模型(VSM):给定一文本D=D(,………，）由于在文本中既可以重复出现又应该有先后次序的关系,分析起来有一定困难。为了简化分析,暂时不考虑的顺序,并要求互异,这时可以把,………，看作是一个n维的坐标,而就是n维坐标所对应的值,所以文档D()就可以被看作一个n维的向量了。

（5） 相似度(Similarity)两个文本D,和DZ之间的(内容)相关程度(Degree of Relevance)常常用他们之间的相似度Sim(,)来度量,当文本被表示为向量空间模型时,我们可以借助与向量之间的某种距离来表示文本间的相似度"常用向量之间的内积进行计算或者用夹角的余弦值表示。



#### Latent Space Models

一种用于信息检索中的常见方法，可以将查询和文档同时映射到一个高维的语义空间中，从而进行匹配。

在 Sco Deerwester, Susan T Dumais, George W Furnas, omas K Landauer, and Richard Harshman. 1990. Indexing by latent semantic analysis. Journal of the American society for information science 41, 6 (1990), 391. 中，作者提出了 Latent Semantic Indexing (LSI)，通过对词频语料库的奇异值分解构造单词和文档的隐层空间向量。

在 Tomas Mikolov, Kai Chen, Greg Corrado, and Jerey Dean. 2013. Ecient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781 (2013). 中，作者通过基于深度神经网络的分布式表示学习方法提出了 word2vec 模型，可以以较低的运算开销学习语料库中的单词向量表示。



#### 分布式表示学习（distributed representation learning）

在 NLP（自然语言处理）中，局部表示更多被称为 one-hot representation，分布式词表示通常被称为词向量或词嵌入 word embedding。

1. Latent Semantic Analysis（LSA）
2. Global Vector（GloVe）
3. Word2vec
4. paragraph vector models （https://arxiv.org/abs/1405.4053）



先前的类似工作有 Qingyao Ai, Liu Yang, Jiafeng Guo, and W Bruce Cro. 2016. Analysis of the paragraph vector model for information retrieval. In Proceedings of the ACM ICTIR’16. ACM, 133–142. 提出的 paragraph vector model，如下图

![1](/Users/hansen/Downloads/Final_Project/Assets/1.jpg)

它通过极大似然估计的目标函数进行优化，从而同时学习 word 和 document 在向量空间中的联合表示。



### 个性化产品搜索的研究方法

有关个性化产品搜索的现有方法仅仅将用户信息用作检索模型中的额外特征，并无差别地在所有搜索会话中个性化设置。 这些研究仅仅使用与查询无关的信息来构造用户资料（如评论和购买历史记录），而对于上文提到的挑战，即何时进行个性化检索来提高产品搜索的性能，却几乎没有提供任何见识。

#### HEM (hierarchical embedding model)

模型由 Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, and W Bruce Croft. 2017. Learning a hierarchical embedding model for personalized product search. In Proceedings of the 40th International ACM SIGIR. ACM, 645–654. 首先提出。此模型基于分布式表示学习（distributed representation learning），将查询，用户信息，产品信息通过深度神经网络训练，映射到同一个向量空间（vector space）中，从而通过向量之间的相似性度量对候选产品进行排序，得到最终的个性化产品搜索结果。

文中通过 Latent Semantic Space 方式实现个性化产品搜索，将用户和查询的信息合并表达为一个向量，然后与表达产品的向量进行相似性度量，如内积或者夹角的余弦值。



## Abstract



------



[TOC]

------



## Introduction

### motivation

Due to the increasing popularity of online shopping and a large number of products on e-commerce websites, product search has become one of the most popular methods for customers to discover products online. In a typical product search scenario, a user would first issue a query on the e-commerce website to get a list of relevant products, then browse the result page, and select one or more items to purchase. Therefore, the quality of product search results has a direct impact on customer satisfaction and the number of transactions on e-commerce websites.

### Issue

When and how to conduct personalization is an important research question for product search in practice.

This is because that search personalization has been shown to potentially have negative effects by previous studies on Web search (4,13,38)

```
[4] Paul N Bennett, Ryen W White, Wei Chu, Susan T Dumais, Peter Bailey, Fedor Borisyuk, and Xiaoyuan Cui. 2012. Modeling the impact of short-and long-term behavior on search personalization. In Proceedings of the 35th international ACM SIGIR. ACM, 185–194.
[13] Susan T Dumais. 2016. Personalized Search: Potential and Pitfalls.. In CIKM. 689.
[38] Jaime Teevan, Susan T Dumais, and Daniel J Liebling. 2008. To personalize or not to personalize: modeling queries with variation in user intent. In Proceedings of the 31st ACM SIGIR. ACM, 163–170.
```

It is possible that incorporating unreliable personal information could exacerbate the problem of data sparsity and introduce unnecessary noise into a search model.

### Solution

#### 1. [A Zero Attention Model for Personalized Product Search](https://arxiv.org/abs/1908.11322)



### Structure 



------



## Related Work

### Latent semantic models

https://en.wikipedia.org/wiki/Latent_semantic_analysis

- Neural Embedding Models
- Hierarchical Embedding Model
- 

------



## Methods

### PRELIMINARY ANALYSIS of 2 conditions for personalization

1. Personalization is beneficial only when the query carries diverse purchase intents.
2. Personalization is beneficial only when the personal preferences of individuals are significantly different from their aggregated group preference.

### Zero Attention Model (ZAM)

Constructing user profiles as a weighted combination of their previously purchased items, to automatically determine when and how to personalize search results based on the current query and user information.



------



## Implementation & Results Analysis

### Dataset

### Evaluation Metrics



------



## Conclusion

1. Personalization appears to be useful for queries with medium or high frequency, it tends to be less beneficial on tail queries with low frequency. 
2. The importance of personalization in product search often depends on the interactions between query context and the user’s previous purchases.







