import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def load(path, data_type):
    """
    path:数据集路径
    type:train或test
    根据指定路径读取训练集或测试集
    """
    data = scipy.io.loadmat(path)

    # 原始数据的label是0/1格式,需要转化为课上学的-1/1格式
    # unit8->int 0/1->-1/1
    if data_type == 'train':
        data['y'] = data['y'].astype(np.int) * 2 - 1
    elif data_type == 'test':
        data['ytest'] = data['ytest'].astype(np.int) * 2 - 1

    return data

def func(train_x, train_y, W, b, lambda_, loss_type):
    """
    根据当前W和b,计算训练集样本的目标函数平均值
    """
    num_train = train_x.shape[0]
    func_ = 0
    for i in range(num_train):
        x_i = train_x[[i]].T  # 1899*1
        y_i = train_y[i]  # 1
        if loss_type == 'hinge':
            func_ += max(0, (1 - y_i * (np.dot(W.T, x_i) + b)))
        elif loss_type == 'exp':
            func_ += np.exp((-y_i * (np.dot(W.T, x_i) + b)))
        elif loss_type == 'log':
            func_ += np.log(1 + np.exp((-y_i * (np.dot(W.T, x_i) + b))))

    func_ /= num_train
    func_ += (lambda_ / 2) * ((np.linalg.norm(W, ord=2)) ** 2)
    return func_[0][0]

def plot(func_list_best, func_interval, loss_type, C, T, acc_best):
    """
    绘制在测试集上结果最佳的模型在训练过程中的目标函数曲线
    """
    ts = [t for t in range(0, T, func_interval)]
    plt.plot(ts, func_list_best, 'k', label='training_cost')
    plt.title('{} acc={}% C={} T={}'.format(loss_type, acc_best, C, T))
    plt.xlabel('t')
    plt.ylabel('f(W,b)')
    plt.savefig('./output/{} acc={} C={} T={}.jpg'.format(loss_type, acc_best, C, T))

def pegasos(train, test, C, T, loss_type='hinge', func_interval=100):

    train_x = train['X']  # 4000*1899
    train_y = train['y']  # 4000*1

    test_x = test['Xtest']  # 1000*1899
    test_y = test['ytest']  # 1000*1

    num_train = train_x.shape[0]  # 4000
    num_test = test_x.shape[0]  # 1000
    num_features = train_x.shape[1]  # 1899

    lambda_ = 1 / (num_train * C)

    # 高斯初始化权重W和偏置b
    W = np.random.randn(num_features, 1)  # 1899*1
    b = np.random.randn(1)

    func_list = []

    # 随机生成一组长度为T,元素范围在[0, num_train-1]的下标,供算法中随机选取训练样本
    choose = np.random.randint(0, num_train, T)

    for t in range(1, T+1):
        # TODO:写出eta_t的计算公式
        # 下降步长，逐渐减小
        eta_t = 1/(lambda_*t)

        # 随机选取的训练样本下标
        i = choose[t-1]
        x_i = train_x[[i]].T  # 1899*1
        y_i = train_y[i]  # 1

        if loss_type == 'hinge':
            # TODO:写出hinge_loss下的梯度更新公式
            # w: 1899 * 1  x: 1899 * 1
            st = y_i * (np.dot(W.T, x_i) + b)
            if st < 1:
                W = W - eta_t * (lambda_ * W - y_i * x_i)
                b = b + eta_t * y_i
            else:
                W = W - eta_t * (lambda_ * W)

        elif loss_type == 'exp':
            exponent = -y_i * (np.dot(W.T, x_i) + b)[0]
            if exponent < 3:

                # TODO:写出exp_loss下的梯度更新公式

                W = W - eta_t * (lambda_ * W - y_i * x_i * np.exp(exponent))
                b = b + eta_t * y_i * np.exp(exponent)

        elif loss_type == 'log':
            # TODO:写出log_loss下的梯度更新公式
            exponent = -y_i * (np.dot(W.T, x_i) + b)[0]
            if exponent < 3:

                W = W - eta_t * (lambda_ * W + (- y_i * x_i * np.exp(exponent)) /(1 + np.exp(exponent)))
                b = b + eta_t * (y_i * np.exp(exponent)) / (1 + np.exp(exponent))

        t += 1

        # 根据当前W和b,计算训练集样本的目标函数平均值
        if t % func_interval == 0:
            func_ = func(train_x, train_y, W, b, lambda_, loss_type)
            func_list.append(func_)
            print('t = %d, func = %.4f' % (t, func_))

    accuracy = 0
    for i in range(test_x.shape[0]):
        res = np.dot(W.T, test_x[i].T) + b
        if res >= 0 and test_y[i] == 1:
            accuracy += 1
        elif res < 0 and test_y[i] == -1:
            accuracy += 1

    accuracy = 100 * accuracy / num_test
    print('accuracy = %.1f%%' % accuracy)

    return accuracy, func_list


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--C', default=0.001, type=float)
    parser.add_argument('-T', '--T', default=10000, type=int)
    parser.add_argument('-l', '--loss', default=2, type=int)
    args = parser.parse_args()

    C = args.C
    T = args.T

    print(args.C, args.T, args.loss)

    # C = 0.001
    # T = 10000  # 迭代次数
    func_interval = 500  # 每隔多少次迭代计算一次目标函数
    times = 4  # 测试次数

    # loss类型切换
    loss_types = ['hinge', 'exp', 'log']
    # loss_type = loss_types[2]
    loss_type = loss_types[args.loss]

    train = load('./data/spamTrain.mat', 'train')  # 4000条
    test = load('./data/spamTest.mat', 'test')  # 1000条

    stat_acc = []
    acc_best = 0  # 选取最好的准确率画图
    func_list_best = []
    for i in range(times):
        print('times:%d' % (i + 1))

        acc, func_list = pegasos(train, test, C, T, loss_type, func_interval)

        if acc > acc_best:
            acc_best = acc
            func_list_best = func_list

        stat_acc.append(acc)

    print('stat_acc', stat_acc)
    print('acc_best', acc_best)
    print('func_list_best', func_list_best)

    plot(func_list_best, func_interval, loss_type, C, T, acc_best)


