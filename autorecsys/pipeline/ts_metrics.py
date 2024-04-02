import numpy as np

def ndcg_(data_topk, pred_topk, ovrlp):
    data_topk = data_topk.reset_index(drop=True)
    data_topk.index += 1

    pred_topk.loc[:, 'rel'] = 0
    pred_topk = pred_topk.reset_index(drop=True)
    pred_topk.index += 1

    for i in ovrlp:
        a = data_topk.movieID[data_topk.movieID == i].index
        b = pred_topk.movieID[pred_topk.movieID == i].index
        pred_topk.loc[b, 'rel'] = a

    pred_topk['true'] = pred_topk.index[::-1]
    nDCG = ndcg_score(pred_topk['true'].values, pred_topk['rel'].values)

    return round(nDCG, 4)

def MRR_(data_topk, ovrlp):
    # MRR
    allindex = []
    ovrlp_ = data_topk.movieID
    ovrlp_ = ovrlp_.reset_index(drop=True)
    ovrlp_.index += 1

    for i in ovrlp:
        n = ovrlp_[ovrlp_ == i].index.values
        allindex.append(n.min())

    if allindex:
        mrr_ = min(allindex)
        mrr_ = 1 / mrr_
    else:
        mrr_ = 0

    return mrr_



##
def dcg_score(data):
    # data: np.array
    gain = 0
    for i in range(len(data)):
        discounts = data[i] / np.log2(i + 2)
        gain = gain + discounts
    return gain


def ndcg_score(true, pred):
    # true, pred: np.array
    iDCG = dcg_score(true)
    DCG = dcg_score(pred)
    return DCG / iDCG
