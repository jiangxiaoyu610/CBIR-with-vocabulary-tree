"""
本代码为词汇树的类。
词汇树本质上是利用树结构进行层级迭代 k-means 聚类。
"""
import time
import numpy as np

from tree_structure import Tree
from sklearn.cluster import KMeans, MiniBatchKMeans


class VocabularyTree:
    def __init__(self, depth, k_per_level):
        self.root = Tree()
        self.depth = depth
        self.k_per_level = k_per_level
        self.num_of_words = self.k_per_level ** (self.depth-1)
        self.n_tree_nodes = int(((self.k_per_level ** self.depth) - 1) / (self.k_per_level - 1))

        self.leaf_nodes = []
        self.center_descriptors = {}

    def hierarchical_clustering(self, descriptors, logger):
        """
        依托树结构进行层级迭代聚类。
        :param descriptors:
        :param logger:
        :return:
        """
        mission = [(0, self.root)]
        indices_dict = {'root': np.arange(len(descriptors))}
        # descriptors = np.array(descriptors)

        # 确认当前层是否还有需要处理的结点
        count = 0
        while mission:
            begin = time.time()
            cur_deep, cur_node = mission.pop(0)

            if cur_deep + 1 < self.depth:
                cur_indices = indices_dict.pop(cur_node.val)
                if len(cur_indices) < self.k_per_level:
                    logger.write("node {} just has {} samples, put it to leaf node!".format(cur_node.val, len(cur_indices)))
                    self.leaf_nodes.append(cur_node.val)
                    continue

                cur_samples = np.take(descriptors, cur_indices, axis=0)

                k_means = MiniBatchKMeans(self.k_per_level, batch_size=100000)
                k_means.fit(cur_samples)
                labels = k_means.predict(cur_samples)

                # 处理聚类结果
                for num in range(self.k_per_level):
                    # 设置树节点编号。例如：‘0_2’代表第一层第0个结点下的第2个结点
                    if cur_node.val == 'root':
                        node_id = str(num)
                    else:
                        node_id = cur_node.val + '_' + str(num)

                    next_node = Tree(node_id)
                    cur_node.add_child(next_node)
                    self.center_descriptors[node_id] = k_means.cluster_centers_[num]

                    # 如果当前深度未达到要求，则将下一结点加入待处理序列。
                    mission.append((cur_deep+1, next_node))
                    tmp_indices = cur_indices[labels == num]
                    indices_dict[node_id] = tmp_indices
            else:
                self.leaf_nodes.append(cur_node.val)

            count += 1
            logger.write("finish {}/{} nodes, used {} seconds".format(count, self.n_tree_nodes, time.time()-begin))

        if len(self.leaf_nodes) == 0:
            raise ValueError("Maybe your tree depth and branch is not proper the few samples."
                             "The tree has not leaf nodes!")

        return

    def get_word_vector(self, image_descriptors):
        """
        生成图片对应的词向量。
        :param image_descriptors:
        :return:
        """
        word_vector = np.zeros((len(self.leaf_nodes)))

        for descriptor in image_descriptors:
            cur_node = self.root
            while cur_node.children:
                min_distance = float('inf')
                min_distance_node_id = None
                for node_id, node in cur_node.children.items():
                    center = self.center_descriptors[node_id]
                    distance = np.linalg.norm(descriptor-center)
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_node_id = node_id

                cur_node = cur_node.children[min_distance_node_id]

            word_vector[self.leaf_nodes.index(cur_node.val)] += 1

        return word_vector
