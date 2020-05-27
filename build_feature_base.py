"""
本代码用于利用SIFT、BOW、k-means、“词汇树”等技术进行特征库的建造。
"""
import time
import random

from common import *
from logger import Logger
from sklearn import preprocessing
from vocabulary_tree import VocabularyTree

# 对于 oxbuild 数据集来说树节点最多100万左右。
DEPTH = 7
K_PER_LEVEL = 10
TRAIN_SET_RATE = 0.002  # 用图片文件夹中的多少比例来构建树。

random.seed(10)


def get_train_data(percent=1.0):
    """
    提取用于构建特征库的图片。
    :param percent: 提取总体数据集的百分比
    :return:
    """
    files_list = []
    for file in os.listdir(DATA_FOLDER):
        files_list.append(DATA_FOLDER + file)

    random.shuffle(files_list)
    index = int(len(files_list) * percent)

    return files_list[:index]


def get_extractor(method):
    """
    生成算子提取器
    :param method:
    :return:
    """
    if method == 'sift':
        extractor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        extractor = cv2.xfeatures2d.SURF_create()
    elif method == 'orb':
        extractor = cv2.ORB_create()
    else:
        raise ValueError

    return extractor


def generate_descriptors(files_list, logger, method='sift'):
    """
    提取每张图片的 SIFT 特征，并拼接在一起返回特征矩阵。
    :param files_list:
    :param logger:
    :param method:'sift', 'surf' or 'orb'
    :return:
    """
    descriptors_file = FEATURE_FOLDER + 'image_descriptor_percent_{}.pkl'.format(TRAIN_SET_RATE)
    if os.path.exists(descriptors_file):
        image_descriptors_dict = load_data(descriptors_file)
        logger.write("finish loading {}".format(descriptors_file))
        return image_descriptors_dict

    image_descriptors_dict = {}
    extractor = get_extractor(method)

    logger.write('\nbegin extract image feature:')
    for i, image_path in enumerate(files_list):
        begin = time.time()
        file = os.path.basename(image_path)
        image_id = file.split('.')[0]

        image = cv2.imread(image_path)

        key_points, descriptors = extractor.detectAndCompute(image, None)
        image_descriptors_dict[image_id] = descriptors

        logger.write("finish extract {}/{} image feature, used: {}".format(i+1, len(files_list), time.time()-begin))

    save_data(image_descriptors_dict, descriptors_file)
    logger.write('finish writing: {}'.format(descriptors_file))

    return image_descriptors_dict


def get_image_feature(image_descriptors_dict, vocabulary_tree, logger, idf=None):
    """
    本函数用于提取特征图片。
    生成每幅图片的视觉词向量的 TF-IDF 向量，并做归一化。
    :param image_descriptors_dict:
    :param vocabulary_tree:
    :param logger:
    :param idf:
    :return:
    """
    word_vector_dict = {}
    document_frequency = np.zeros((vocabulary_tree.num_of_words,))

    count, n_images = 0, len(image_descriptors_dict)
    logger.write('begin generate word vector:')
    for image_id, descriptors in image_descriptors_dict.items():
        begin = time.time()
        if descriptors is not None:
            word_vector = vocabulary_tree.get_word_vector(descriptors)
        else:
            word_vector = np.zeros((vocabulary_tree.num_of_words, ))

        word_vector_dict[image_id] = word_vector

        document_frequency += (word_vector > 0) * 1

        count += 1
        logger.write('finish generating {}/{} image word vector, used: {} seconds'.format(count, n_images, time.time()-begin))

    # 计算 TF-IDF
    if idf is None:
        idf = np.log((n_images+1)/(document_frequency + 1))

    for image_id in word_vector_dict.keys():
        word_vector_dict[image_id] *= idf
        word_vector_dict[image_id] = preprocessing.normalize(word_vector_dict[image_id].reshape((1, -1)), norm='l2')
        word_vector_dict[image_id] = word_vector_dict[image_id].reshape((-1, ))

    return word_vector_dict, idf


def build_vocabulary_tree(image_descriptors_dict, logger):
    """
    建立词汇树，利用树结构进行层级循环式k-means
    :param image_descriptors_dict:
    :param logger:
    :return:
    """
    descriptors_list = None
    for descriptors in image_descriptors_dict.values():
        if descriptors is not None:
            descriptors = descriptors.astype(np.int32)
            if descriptors_list is None:
                descriptors_list = descriptors
            else:
                descriptors_list = np.concatenate((descriptors_list, descriptors), axis=0)

    logger.write('\nbegin building vocabulary tree:')
    logger.write('samples: {}'.format(len(descriptors_list)))

    vocabulary_tree = VocabularyTree(DEPTH, K_PER_LEVEL)
    vocabulary_tree.hierarchical_clustering(descriptors_list, logger)

    return vocabulary_tree


def save_related_data(vocabulary_tree, word_vector_dict, idf):
    """
    储存词汇树、图片特征、IDF 系数
    :param vocabulary_tree:
    :param word_vector_dict:
    :param idf:
    :return:
    """
    if not os.path.exists(FEATURE_FOLDER):
        os.mkdir(FEATURE_FOLDER)

    vocabulary_tree_file = FEATURE_FOLDER + '{}_{}_{}_vocabulary_tree.pkl'.format(DEPTH, K_PER_LEVEL, int(TRAIN_SET_RATE))
    save_data(vocabulary_tree, vocabulary_tree_file)

    idf_file = FEATURE_FOLDER + '{}_{}_{}_inverse_document_frequency.pkl'.format(DEPTH, K_PER_LEVEL, int(TRAIN_SET_RATE))
    save_data(idf, idf_file)

    image_feature_file = FEATURE_FOLDER + '{}_{}_{}_image_feature.pkl'.format(DEPTH, K_PER_LEVEL, int(TRAIN_SET_RATE))
    save_data(word_vector_dict, image_feature_file)

    return


def process():
    """
    本代码用于建造图像检索系统的特征库。
    具体步骤如下：
        1）提取每张图片的 SIFT 特征。
        2）建立词汇树，利用树结构进行层级循环式k-means
        3）生成每幅图片的视觉词 TF-IDF 向量，并做归一化。
        4）储存每幅图片的特征向量。
    :return:
    """
    logger = Logger('./logs/{}_log.txt'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))

    files_list = get_train_data(TRAIN_SET_RATE)
    image_descriptors_dict = generate_descriptors(files_list, logger)

    vocabulary_tree = build_vocabulary_tree(image_descriptors_dict, logger)

    word_vector_dict, idf = get_image_feature(image_descriptors_dict, vocabulary_tree, logger)

    save_related_data(vocabulary_tree, word_vector_dict, idf)

    return


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--depth', type=int, help="depth of vocabulary tree")
    parse.add_argument('--branch', type=int, help="branch of vocabulary tree")
    parse.add_argument('--train_set_rate', type=float, help='percent of usage of data folder')

    args = parse.parse_args()
    DEPTH = args.depth
    K_PER_LEVEL = args.branch
    TRAIN_SET_RATE = args.train_set_rate

    process()
