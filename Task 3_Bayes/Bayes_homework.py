import copy
import nltk
import string
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# 种类
categories = {'World': 0, 'Sci/Tech': 1, 'Sports': 2, 'Business': 3}
# 停用词
stopwords = set(nltk.corpus.stopwords.words('english'))
# 词干提取/词形还原
stemmer, lemmatizer = PorterStemmer(), WordNetLemmatizer()


def preprocess(sent, type_word):
    """
    文本预处理函数,将输入的句子转化为单词词组,并统一为小写、去标点、去停用词、去数字、还原
    """

    # 统一为小写
    sent = sent.lower()
    # 去标点
    remove = str.maketrans('', '', string.punctuation)
    sent = sent.translate(remove)
    # 转化为单词词组
    words = nltk.word_tokenize(sent)
    # 去停用词
    words = [w for w in words if not (w in stopwords)]
    # 去数字
    words = [w for w in words if not w.isdigit()]
    # 还原:词干提取/词形还原
    if type_word == 'stemmer':
        words = [stemmer.stem(w) for w in words]

    elif type_word == 'lemmatizer':
        words = [lemmatizer.lemmatize(w) for w in words]

    return words

def load(path, type_word):
    """
    path:数据集路径
    根据指定路径读取训练集或测试集
    """
    data_x, data_y = [], []
    with open(path, 'r') as f:
        lines = f.readlines()
        length = len(lines)
        for i, line in enumerate(lines):
            temp = line.split('|')
            data_x.append(preprocess(temp[1].strip(), type_word))
            data_y.append(temp[0])
            if i % 1000 == 0:
                print('loading:{}/{}'.format(i, length))

    return data_x, data_y

def words2dic(train_x):
    """
    将训练集中的单词转化为词-id(从0开始)的字典
    """
    dictionary = {}
    i = 0
    for words in train_x:
        for word in words:
            if not word in dictionary:
                dictionary[word] = i
                i += 1
    return dictionary

def train_TF(train_x, train_y):
    # train_x, train_y: (10208)
    # 18359
    dictionary = words2dic(train_x)
    # 词-类-词频矩阵(维度:词典长度x类别数)
    # (18359, 4)
    words_frequency = np.zeros((len(dictionary), len(categories)), dtype=int)
    for i, words in enumerate(train_x):
        # 'Business', 'Sports', 'Sci/Tech', 'World'
        category_id = categories[train_y[i]]
        for word in words:
            word_id = dictionary[word]
            # 每个类别，每个词出现的次数
            words_frequency[word_id][category_id] += 1

    # 每类下的句总数(维度:类别数x1)
    # 每个类的样本数
    # 2521,2619,2531,2537
    category_sents = np.zeros(len(categories))
    for category in train_y:
        category_id = categories[category]
        category_sents[category_id] += 1
    p_c = category_sents / len(train_y)

    # 每类下的词总数(维度:类别数x1)

    # 49504,50056,48022,49811
    category_words = np.sum(words_frequency, axis=0)

    # 统计矩阵(维度:词典长度x类别数)
    # (18359, 4)
    # p_stat = np.zeros((len(dictionary), len(categories)))
    # TODO:计算TF方法下的P(w_i|c),即这里的p_stat

    p_stat = (words_frequency + 1) / (category_words + len(dictionary))

    return p_stat, dictionary, p_c

def train_Bernoulli(train_x, train_y):
    dictionary = words2dic(train_x)
    # 词-类-句总数矩阵(维度:词典长度x类别数)
    sents_count = np.zeros((len(dictionary), len(categories)), dtype=int)
    for i, words in enumerate(train_x):
        category_id = categories[train_y[i]]
        for word in set(words):
            word_id = dictionary[word]
            # 包含 x (word) 的文档数目
            sents_count[word_id][category_id] += 1

    # 每类下的句总数(维度:类别数x1) 属于 wj 类的文档总数
    category_sents = np.zeros(len(categories))
    for category in train_y:
        category_id = categories[category]
        category_sents[category_id] += 1
    # 先验概率
    p_c = category_sents / len(train_y)

    # 统计矩阵(维度:词典长度x类别数)
    # p_stat = np.zeros((len(dictionary), len(categories)))
    # TODO:计算Bernoulli方法下的P(d_?|c),即这里的p_stat

    p_stat = (sents_count + 1) / (category_sents + 2)

    return p_stat, dictionary, p_c

def test_TF(data_x, data_y, p_stat, dictionary, p_c):
    """
    批量数据测试,计算准确率
    """
    # 计算ln P(C)
    p_c = np.log(p_c)

    # 分类正确的个数
    count = 0
    # 计算P,即argmax之后的内容
    for i, words in enumerate(data_x):
        p = np.zeros(len(categories))
        # TODO:计算TF方法下的ln P(text|c)P(c),即这里的p
        for word in words:
            if word in dictionary:
                word_id = dictionary[word]
                p += np.log(p_stat[word_id,:])
        p += p_c

        if np.argmax(p) == categories[data_y[i]]:
            count += 1

    print('Accuracy: {}/{} {}%'.format(count, len(data_y), round(100*count/len(data_y), 2)))

def test_Bernoulli(data_x, data_y, p_stat, dictionary, p_c):
    """
    批量数据测试,计算准确率
    """
    # 计算ln P(C)
    p_c = np.log(p_c)

    # 分类正确的个数
    count = 0
    # 计算P,即argmax之后的内容
    for i, words in enumerate(data_x):
        # p = np.zeros(len(categories))
        # TODO:计算Bernoulli方法下的ln P(text|c)P(c),即这里的p
        b = np.zeros(len(dictionary))
        for word in set(words):
            if word in dictionary:
                word_id = dictionary[word]
                b[word_id] = 1

        b = np.expand_dims(b, 1).repeat(len(categories), 1)
        _b = np.ones(b.shape) - b

        p = np.log(p_stat) * b + np.log(1 - p_stat) * _b
        p = np.sum(p, axis=0)
        p += p_c
        if np.argmax(p) == categories[data_y[i]]:
            count += 1

    print('Accuracy: {}/{} {}%'.format(count, len(data_y), round(100*count/len(data_y), 2)))

# def use(sent, p_stat, dictionary, p_c, type_word, type_train):
#     """
#     针对单个句子输出分类概率
#     """
#     words = preprocess(sent, type_word)
#
#     # 计算ln P(C)
#     p_c = np.log(p_c)
#     p = []
#
#     # 计算P
#     if type_train == 'TF':
#         p = copy.deepcopy(p_c)  # ln P(C)
#         for word in words:
#             # 防止未登录词
#             if word in dictionary:
#
#
#
#     elif type_train == 'Bernoulli':
#         p = copy.deepcopy(p_c)  # ln P(C)
#         temp = 1 - copy.deepcopy(p_stat)  # 1-P(Word|C)
#         for word in set(words):
#             # 防止未登录词
#             if word in dictionary:
#
#                 for category in categories:

    # softmax打印概率
    # p = np.exp(p)
    # print(sent)
    # print(' | '.join(list(categories.keys())))
    # np.set_printoptions(suppress=True)
    # print(100*p/np.sum(p, axis=0, keepdims=True))



if __name__ == '__main__':

    type_word = ['stemmer', 'lemmatizer'][0]
    # 训练方法
    type_train = ['TF', 'Bernoulli'][0]

    train_x, train_y = load('./data/news_category_train_mini.csv', type_word)
    test_x, test_y = load('./data/news_category_test_mini.csv', type_word)

    print(type_word)
    print(type_train)

    if type_train == 'TF':
        p_stat, dictionary, p_c = train_TF(train_x, train_y)
        # 训练集上的准确率
        test_TF(train_x, train_y, p_stat, dictionary, p_c)
        # 测试集上的准确率
        test_TF(test_x, test_y, p_stat, dictionary, p_c)

    elif type_train == 'Bernoulli':
        p_stat, dictionary, p_c = train_Bernoulli(train_x, train_y)
        # 训练集上的准确率
        test_Bernoulli(train_x, train_y, p_stat, dictionary, p_c)
        # 测试集上的准确率
        test_Bernoulli(test_x, test_y, p_stat, dictionary, p_c)

    # sents = ["India and Pakistan were unable to agree on starting a bus service between the capitals of divided Kashmir in two days of talks which ended in New Delhi on Wednesday and said they would need more discussions.",
    #          "Alternative Web browsers Mozilla and FireFox experienced another month of growth at the expense of Microsoft #39;s dominant Internet Explorer, according to an online study.",
    #          "Iraq's footballers extended their fairytale run at the Athens Olympics Saturday, beating  Australia 1 0 to reach the semi finals of the men's tournament.",
    #          "Crude oil futures rose to a record for a third day Monday, surpassing  $55 in New York on speculation US demand for heating oil will deplete inventories this winter."]
    #
    # for sent in sents:
    #     use(sent, p_stat, dictionary, p_c, type_word, type_train)
