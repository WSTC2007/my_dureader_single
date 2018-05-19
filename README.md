# my_dureader

运行环境：python 3.5/3.6，pytorch 0.2.0，自行安装fasttext

参考百度给的tensorflow基线代码，改成了pytorch版本

1. 该版本实现的是match-lstm，bidaf可以后期实现
2. 和基线代码的不同点：
   - match层的输入不同
   - pointer network的初始化隐层向量不同
   - 如果结果差太多，看看是否是不同框架之间参数初始化不同
   - 词向量初始化采用fasttext
   - 训练的时候采用单篇文章方式，测试的时候选range概率最大的答案


