# Ant Financial Question Matching Corpus
- [AFQMC](https://tianchi.aliyun.com/dataset/106411) (Ant Financial Question Matching Corpus) ：蚂蚁金融语义相似度数据集，该数据集由蚂蚁金服提供。

# Classfication
- [bert-base-chinese](https://huggingface.co/bert-base-chinese)
- [CLS] => Linear(hidden_size, 1) => nn.NCELoss()

# HyperParameter
- "batch_size": 32,
- "lr": 1e-4,
- "weight_decay": 0.01,
- "epoch": 10

# Result
- ACC: 73.1%
- auc: 0.7683
- Run once on 8-V100

![截屏2024-01-28 20 36 40](https://github.com/bigjeager/afqmc/assets/60964665/520e1965-8cda-4d55-94d1-01c145fc4c3c)

# Tips
- loss.mean() will be affected by large loss values, which will led to the "fake" loss explosion
- use loss.median() in validation

<img width="1291" alt="截屏2024-01-28 20 47 17" src="https://github.com/bigjeager/afqmc/assets/60964665/00addcff-a02c-4c97-8274-0e5321b30781">
<img width="1288" alt="截屏2024-01-28 20 47 29" src="https://github.com/bigjeager/afqmc/assets/60964665/5b0c227b-8e75-4670-b4e6-4a78e0490c01">
