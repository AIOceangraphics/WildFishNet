import numpy as np
import scipy.spatial.distance as spd
import torch

import libmr
# numpy 1.22.1




def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):     # query_score, mcv, eu_weight 求得（欧几里得+余弦）距离

    if distance_type == 'eucos':                                           # 从CNN的倒数第二层提取正确分类的训练样本的激活向量，根据不同的类别（真实label）将这些向量对应分开，
                                                                           # 然后分别计算每个类别对应向量的均值作为该类别的中心。然后分别计算每个类别中每个样本对应向量和其类别中心的距离，
                                                                           # 然后对这些距离进行排序，针对排序后的几个尾部极大值进行极大值理论分析，
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + \
            spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)     #欧式距离
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)        #计算余弦相似性，余弦距离
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):   # 极大值的分布符合weibull分布，所以使用weibull分布（libmr中的fithigh方法）来拟合这些极大的距离，
                                                                                 # 得到一个拟合分布的模型
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))     # 给定序列a=[1,2,3....1000]和序列b=[1000,999,....1]不论是fit_high(a, tail_size=20)
                                                       # 还是fit_high(b, tail_size=20)拟合的都是[1000,999...981]这些值的Weibull分布。
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))

        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)

    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    modified_scores = scores.tolist() + [unknowns]
    return modified_scores


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_score, eu_weight, alpha=10, distance_type='eucos'):    # weibull_model：拟合模型,
                                                                                                    # categories：训练集的种类列表,
                                                                                                    # input_score,
                                                                                                    # eu_weight=0.5,
                                                                                                    # alpha=10,
                                                                                                    # distance_type='eucos'：距离类型（欧氏距离）
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)

    ranked_list = input_score.argsort().ravel()[::-1][:alpha]           # argsort()函数是对数组中的元素进行从小到大排序，并返回相应序列元素的数组下标。
    alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]    # 计算w的值（该值的计算就是根据采用tail的个数(即阿尔法的值)来计算的，比较简单就是一个固定的式子）
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights

    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)     # 每个向量分别针对每个类别计算与其之间的距离（共得到N（类别数量）个距离）
            wscore = model[channel].w_score(channel_dist)    # 针对上述得到的每个距离分别使用每个类别对应的拟合模型对其进行预测，
                                                             # 得到一个分数FitScores，这个分数就是指得是该测试图像特征值归属于其对应类别的的概率，所以一共是有N个分数的。
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])  # v^i(x)=v(x)*(1-welbull *(alpha-i)/alpha)
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)    # v^0(x)=sum(vi(x)*(1-wi(x)))

        scores.append(score_channel)       # openmax后的分数
        scores_u.append(score_channel_u)   # unknowen的分数

    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)

    openmax_prob = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])     # 欧式距离
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])       # 余弦距离
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +           # 欧式距离*0.5+余弦距离
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_train_score_and_mavs_and_dists(train_class_num,trainloader, device, net, test_classes):   # 注意这里以训练样本进行EVT拟合
    scores = [[] for _ in range(train_class_num)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            onehot_targets_index = [test_classes.index(i) for i in targets]
            targets = torch.LongTensor(onehot_targets_index)
            inputs, targets = inputs.to(device), targets.to(device)

            # this must cause error for cifar10
            feature, outputs = net(inputs)
            # outputs = net(inputs)

            for score, t in zip(outputs, targets):
                # print(f"torch.argmax(score) is {torch.argmax(score)}, t is {t}")
                if torch.argmax(score) == t:      # 求出所有预测正确样本的向量
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))     # 根据不同的类别（真实label）将这些向量对应分开
    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)      # 然后分别计算每个类别对应向量的均值作为该类别的中心
    dists = [compute_channel_distances(mcv, score) for mcv, score in zip(mavs, scores)]     # 计算每个类别中每个样本对应向量和其类别中心的距离
    return scores, mavs, dists

