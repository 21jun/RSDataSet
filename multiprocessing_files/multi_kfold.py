# For assignment 5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import surprise

from surprise import KNNBasic, KNNWithMeans, KNNWithZScore, SVD
from surprise.model_selection import cross_validate, KFold
from sklearn import metrics

from collections import defaultdict

def job(num):
    return num * 2


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def ndcg_at_k_all(predictions, k):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append([est, true_r])
    
    sum_ndcg = 0.
    
    # iterate over users
    for uid in user_est_true.keys():
        tmp = np.array(user_est_true[uid])
        r = tmp[:, 1][np.argsort(-tmp[:, 0])]
        sum_ndcg += ndcg_at_k(r, k)
    
    return sum_ndcg / len(user_est_true.keys())

def train_with_Kfold(algo, data, k=5, verbose=True):
    
    kf = KFold(n_splits=k,)
    
    history = pd.DataFrame(columns=['precision','recall', 'f1', 'NDCG'])
    
    i = 0
    for trainset, testset in kf.split(data):
        # algo 는 fit의 인자로 trainset 객체를 받고,
        algo.fit(trainset)
        predictions = algo.test(testset) # test의 인자로 튜플의 list 를 받는다.
        precisions, recalls = precision_recall_at_k(predictions, k=15, threshold=4)

        P = sum(rec for rec in precisions.values()) / len(precisions)
        R = sum(rec for rec in recalls.values()) / len(recalls)
        F1 = (2 * P * R) / (P + R)
        # NDCG 의 top k rank 는 k=5 사용
        NDCG = ndcg_at_k_all(predictions, k=5)
        
        history.loc[i]=[P, R, F1, NDCG]
        
        if verbose:
            print(f"FOLD: {i}")
            print("precision: ", P)
            print("recall: ",R)
            print("f1: ",F1)
            print("NDCG: ",NDCG)
            print("------")
        
        i +=1
    
    return history

def work(data, k):

    history = {}

    p_history=[]
    r_history=[]
    f1_history=[]
    ndcg_history=[]

    sim_options = {'name':'cosine', 'user_based': True}
    algo = KNNWithMeans(k=k, min_k=1, sim_options=sim_options, verbose=False)
    KNNWithMeans_history = train_with_Kfold(algo, data, 5, False)
    
    p_history.append(KNNWithMeans_history.mean()[0])
    r_history.append(KNNWithMeans_history.mean()[1])
    f1_history.append(KNNWithMeans_history.mean()[2])
    ndcg_history.append(KNNWithMeans_history.mean()[3])

    history[str(k)] = {
        "precision" : p_history,
        "recall"    : r_history,
        "f1"        : f1_history,
        "ndcg"      : ndcg_history
    }

    return history