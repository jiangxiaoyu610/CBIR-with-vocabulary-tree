import time
import matplotlib.pyplot as plt

from common import *
from logger import Logger
from build_feature_base import generate_descriptors, get_image_feature

# 由于展示时每行 3 张图片，因此建议 TOP N 为 3 的倍数。
TOP_N = 3


def load_related_data(depth, k_per_level, train_set_rate):
    """
    读取词汇树、图片特征、IDF 系数
    :return:
    """
    vocabulary_tree_file = FEATURE_FOLDER + '{}_{}_{}_vocabulary_tree.pkl'.format(depth, k_per_level, train_set_rate)
    vocabulary_tree = load_data(vocabulary_tree_file)

    idf_file = FEATURE_FOLDER + '{}_{}_{}_inverse_document_frequency.pkl'.format(depth, k_per_level, train_set_rate)
    idf = load_data(idf_file)

    image_feature_file = FEATURE_FOLDER + '{}_{}_{}_image_feature.pkl'.format(depth, k_per_level, train_set_rate)
    word_vector_dict = load_data(image_feature_file)

    return vocabulary_tree, word_vector_dict, idf


def find_similar_image(retrieve_image_feature, word_vector_dict, top_n=TOP_N):
    """
    将待检索图片与图片库中的图片特征做点积计算距离，提取前 N 个结果返回。
    :param retrieve_image_feature:
    :param word_vector_dict:
    :param top_n:
    :return:
    """
    retrieve_image_feature = np.array(retrieve_image_feature).reshape((1, -1))
    image_id_list = list(word_vector_dict.keys())
    word_vectors = np.array(list(word_vector_dict.values()))

    similarity = np.dot(retrieve_image_feature, word_vectors.T)
    similarity = -similarity.reshape((-1, ))
    ranked_index = np.argsort(similarity)

    similar_image_file = []
    for index in ranked_index[:top_n]:
        similar_image_file.append(image_id_list[index])

    return similar_image_file


def get_n_rows(similar_image_files):
    """
    展示时固定一行三张图片，计算需要多少行。
    :param similar_image_files:
    :return:
    """
    n = len(similar_image_files)
    if n % 3 == 0:
        n_rows = n // 3
    else:
        n_rows = (n // 3) + 1

    return n_rows + 1


def read_image_by_id(image_id, data_folder='./data/', ext='.jpg'):
    """
    通过图片 ID 读取图片。
    :param image_id:
    :param data_folder:
    :param ext:
    :return:
    """
    image_path = data_folder + image_id + ext
    image = cv2.imread(image_path)

    return image


def show_result(image_file, similar_image_files, logger):
    """
    展示图像检索结果。
    :param image_file:
    :param similar_image_files:
    :param logger:
    :return:
    """
    n_rows = get_n_rows(similar_image_files)

    image = cv2.imread(image_file)
    plt.subplot(n_rows, 3, 2)
    plt.imshow(image)
    plt.title("input image", fontsize=8)
    plt.xticks([])
    plt.yticks([])

    for i, image_id in enumerate(similar_image_files):
        similar_image = read_image_by_id(image_id)
        plt.subplot(n_rows, 3, i + 4)
        plt.imshow(similar_image)
        plt.title('Similar image rank: {}'.format(i + 1), fontsize=8, y=-0.25)
        plt.xticks([])
        plt.yticks([])
        logger.write("No.{} similar image is {}".format((i+1), image_id))

    plt.show()
    plt.close()

    return


def process(image_file, depth, k_per_level, train_set_rate):
    """
    本代码用于完成图像检索功能。
    具体步骤如下：
        1）提取待检索图片的视觉词 TF-IDF 向量。
        2）与图片库中的图片特征做点积计算距离，提取前 N 个结果返回。
        3）结果展示。
    注：由于图片的特征向量已经做过 L2 正则化，因此点积相当于计算余弦距离。
    :return:
    """
    logger = Logger('./logs/{}_testing_log.txt'.format(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))

    vocabulary_tree, word_vector_dict, idf = load_related_data(depth, k_per_level, train_set_rate)

    image_descriptors = generate_descriptors([image_file], logger)

    image_feature, _ = get_image_feature(image_descriptors, vocabulary_tree, logger, idf)
    retrieve_image_feature = list(image_feature.values())[0]

    similar_image_files = find_similar_image(retrieve_image_feature, word_vector_dict)

    show_result(image_file, similar_image_files, logger)

    logger.close()
    return


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--image_file', help="image file to be search")
    parse.add_argument('--depth', help="depth of vocabulary tree")
    parse.add_argument('--branch', help="branch of vocabulary tree")
    parse.add_argument('--train_set_rate', help='percent of usage of data folder')

    args = parse.parse_args()

    process(args.image_file, args.depth, args.branch, args.train_set_rate)
