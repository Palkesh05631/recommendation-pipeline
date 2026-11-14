# Model comparison (K=10, relevant = rating >= 4)

| model                       |   precision@10 |   recall@10 |   ndcg@10 |   coverage |   diversity |   rmse |    mae |   n_eval_users |
|:----------------------------|---------------:|------------:|----------:|-----------:|------------:|-------:|-------:|---------------:|
| Popularity baseline         |         0.0774 |      0.0793 |    0.0981 |     0.0428 |      0.7482 | 1.0734 | 0.8588 |            912 |
| SVD collaborative filtering |         0.137  |      0.1632 |    0.1883 |     0.2693 |      0.7145 | 1.1065 | 0.8774 |            912 |