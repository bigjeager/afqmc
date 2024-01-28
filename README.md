# Ant Financial Question Matching Corpus
- [AFQMC](https://tianchi.aliyun.com/dataset/106411) (Ant Financial Question Matching Corpus) ：蚂蚁金融语义相似度数据集，该数据集由蚂蚁金服提供。

# Classfication
- bert-base-chinese
- [CLS] => Linear(hidden_size, 1) => nn.NCELoss()

# HyperParmeter
- "batch_size": 32,
- "lr": 1e-4,
- "weight_decay": 0.01,
- "epoch": 10

# Result
- ACC: 73.1%
- auc: 0.7683

![截屏2024-01-28 20 36 40](https://github.com/bigjeager/afqmc/assets/60964665/520e1965-8cda-4d55-94d1-01c145fc4c3c)
