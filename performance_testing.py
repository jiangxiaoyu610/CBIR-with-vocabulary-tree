"""
本代码用于测试词汇树的性能
"""
import re
import time

from common import *
from logger import Logger
from search_picture import *


def compute_ap(rank_list, pos_set, neg_set):
    """
    利用 Oxford Building 给定的 average precision 计算方法计算 ap。
    参考 Oxford Building 官网文件 compute_ap.cpp。

    :param rank_list: 词汇树查询返回的结果。
    :param pos_set: Oxford Building 给定的 good、ok 文件集合。
    :param neg_set: Oxford Building 给定的 junk 文件集合。
    :return:
    """
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0

    j = 0
    intersect_size = 0.0
    for i in range(len(rank_list)):
        if rank_list[i] in neg_set:
            continue
        if rank_list[i] in pos_set:
            intersect_size += 1

        recall = intersect_size / len(pos_set)
        precision = intersect_size / (j+1)
        ap += (recall - old_recall) * ((old_precision + precision) / 2)
        old_recall = recall
        old_precision = precision
        j += 1

    return ap


def get_query_dict(file_list):
    """
    遍历 query 文件，返回 query 字典。
    :param file_list:
    :return:
    """
    query_dict = {}
    for file in file_list:
        if 'query' not in file:
            continue

        query_id = re.sub('_query.txt', '', file)
        file_path = LABEL_FOLDER + file
        data = load_file(file_path)
        data = data.split(' ')[0]
        query = data[5:]
        query_dict[query_id] = query

    return query_dict


def get_pos_neg_dict(file_list):
    """
    遍历 good、ok、junk文件，返回正例负例样本。
    :param file_list:
    :return:
    """
    def foo(x): return re.sub(r'\n', '', x)

    positive_dict, negative_dict = {}, {}
    for file in file_list:
        if 'good' in file:
            query_id = re.sub('_good.txt', '', file)
            file_path = LABEL_FOLDER + file
            tmp = load_file(file_path, True)
            tmp = map(foo, tmp)
            if positive_dict.get(query_id) is None:
                positive_dict[query_id] = []
            positive_dict[query_id].extend(tmp)

        elif 'ok' in file:
            query_id = re.sub('_ok.txt', '', file)
            file_path = LABEL_FOLDER + file
            tmp = load_file(file_path, True)
            tmp = map(foo, tmp)
            if positive_dict.get(query_id) is None:
                positive_dict[query_id] = []
            positive_dict[query_id].extend(tmp)

        elif 'junk' in file:
            query_id = re.sub('_junk.txt', '', file)
            file_path = LABEL_FOLDER + file
            tmp = load_file(file_path, True)
            tmp = map(foo, tmp)
            if negative_dict.get(query_id) is None:
                negative_dict[query_id] = []
            negative_dict[query_id].extend(tmp)

        else:
            continue

    return positive_dict, negative_dict


def load_query_data():
    """
    读取测试性能所需的相关文件。
    具体为读取 query、good、ok、junk。
    返回 query 为字典形式，具体格式如下：
    {
        query_id: query_name,
        ...
    }
    返回 pos_dict, neg_dict 为字典形式，具体格式如下:
    {
        query_id: [label1, label2, ...],
        ...
    }
    :return:
    """
    file_list = os.listdir(LABEL_FOLDER)
    query_dict = get_query_dict(file_list)
    positive_dict, negative_dict = get_pos_neg_dict(file_list)

    return query_dict, positive_dict, negative_dict


def process(depth, k_per_level, train_set_rate):
    """
    测试词汇树的性能。
    具体步骤如下：
        1）读取 Oxford Building 数据集给定的测试文件
        2）输入相应的 query，查看返回的结果与good、junk、ok 三个集合的相交程度计算 mAP。
    注：此处 junk 并非负例，而是被遮挡超过 75% 的正例，但是鉴于 OB 官网用此计算方法，此处我们保持统一。
    :return:
    """
    logger = Logger('./logs/{}_testing_log.txt'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))

    query_dict, positive_dict, negative_dict = load_query_data()
    vocabulary_tree, word_vector_dict, idf = load_related_data(depth, k_per_level, train_set_rate)

    count, average_precision = 0, []
    for query_id, query in query_dict.items():
        begin = time.time()
        pos_set = positive_dict[query_id]
        neg_set = negative_dict[query_id]

        image_file = DATA_FOLDER + query + '.jpg'

        image_descriptors = generate_descriptors([image_file], logger)

        image_feature, _ = get_image_feature(image_descriptors, vocabulary_tree, logger, idf)
        retrieve_image_feature = list(image_feature.values())[0]

        similar_image_files = find_similar_image(retrieve_image_feature, word_vector_dict)

        ap = compute_ap(similar_image_files, pos_set, neg_set)
        average_precision.append(ap)

        count += 1
        logger.write("finish {}/{} query, used: {} seconds".format(count, len(query_dict), time.time()-begin))

    logger.write("mAP: {}".format(np.mean(average_precision)))

    logger.close()
    return


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--depth', help="depth of vocabulary tree")
    parse.add_argument('--branch', help="branch of vocabulary tree")
    parse.add_argument('--train_set_rate', help='percent of usage of data folder')

    args = parse.parse_args()

    process(args.depth, args.branch, args.train_set_rate)
