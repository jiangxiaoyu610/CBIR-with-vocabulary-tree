如果sift不能用则下载opencv-contrib-python：
pip install opencv-contrib-python==3.4.2.16

在设置树的深度和每层结点时要注意检查数据数量，看数据量能否被一直除。
否则可能在某个结点的剩余数据不足聚类个数。

下载 The Oxford Buildings Dataset 防到 ./data 目录下

https://github.com/jiangxiaoyu610/CBIR-with-vocabulary-tree.git