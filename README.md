### Introduction
Reproduce `Semantic Product Search` (https://arxiv.org/abs/1907.00937) with many advances for tiki products.

### Steps
`cat_aggs > generate_impressions > make_metadata`


scp -r db/ ubuntu@3.83.35.218:~/semranker/
scp -r meta/  ubuntu@3.83.35.218:~/semranker/
scp -r train_csv/  ubuntu@3.83.35.218:~/semranker/
scp -r transform_impressions/  ubuntu@3.83.35.218:~/semranker/
conda create --name tf-1.13.1 tensorflow-gpu==1.13.1 python=3.6.7

https://github.com/shu-yusa/tensorflow-mirrored-strategy-sample/blob/master/cnn_mnist.py

grep -Ril "text-to-find-here" /