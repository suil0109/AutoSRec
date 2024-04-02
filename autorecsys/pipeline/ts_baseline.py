import pandas as pd
import numpy as np
from autorecsys.pipeline.ts_metrics import MRR_, ndcg_
import random

class NoDNN_topk():
    def __init__(self, baseline, data, topk, timestep=None):
        # pred has ratings only
        # data has all x and ys
        # self.pred = pred
        self.baseline = baseline
        self.data = data
        self.topk = topk
        self.timestep = timestep

    def HR_MRR_nDCG(self):
        # pred = pd.DataFrame(self.pred.flatten(), columns=['pred'])
        data = self.data[['rating', 'userID', 'movieID', 'releaseDate']]

        # disregard the first timestep
        data = data.iloc[self.timestep:]
        data = data.reset_index(drop=True)
        # print(data.shape, pred.shape, 'data')
        # combined = pd.concat([pred, data], axis=1)
        combined = data

        list_a = combined['userID'].unique()
        # get most interactions
        pop_list = combined.movieID.value_counts().head(self.topk*5)
        rndm_list = combined.movieID.value_counts().head(self.topk*10)
        accuracy = []
        mrr = []
        nDCGs = []

        # important: reverse the order
        # order from old -> new to new -> old, so that pd sort the newest first
        tmp = combined[::-1]
        tmp = tmp.groupby(["userID"])

        y = 0
        for x in list_a:
            tmp_ = tmp.get_group(x)
            # pred = tmp_.sort_values(by=['pred'], ascending=False)
            true = tmp_.copy()
            mostpop = tmp_.copy()
            randompop = tmp_.copy()

            # data = tmp_.sort_values(by=['rating'], ascending=False)
            # mostpop.index is the movieID

            if self.baseline == 'mostpop':
                pop_topk = self.MOSTPOP(data, mostpop, pop_list)
            else:
                pop_topk = self.RANDOMPOP(data, randompop, rndm_list)
            true_topk = true.head(self.topk)
            # pd.set_option('display.max_columns', None)
            # pd.set_option('display.max_rows', None)
            print('random')
            print(pop_topk)
            print(true_topk)

            # Most_Pop
            accuracy.append(len(np.intersect1d(true_topk.movieID.values, pop_topk.movieID.values)) / len(true_topk))

            # MRR
            ovrlp = np.intersect1d(true_topk.movieID.values, pop_topk.movieID.values)
            ovrlp = list(ovrlp)
            mrr_ = MRR_(true_topk, ovrlp)
            mrr.append(mrr_)

            # nDCG
            nDCG_ = ndcg_(true_topk, pop_topk, ovrlp)
            nDCGs.append(nDCG_)

        # Accuracy
        print('random')
        avgacc = round((sum(accuracy) / len(accuracy)), 4)
        print('accuracy: ', accuracy)
        print('avg accuracy: ', avgacc)
        # MRR
        avgmrr = round((sum(mrr) / len(mrr)), 4)
        print('mrr: ', mrr)
        print('avg mrr: ', avgmrr)
        # nDCG
        avgnDCG = round((sum(nDCGs) / len(nDCGs)), 4)
        print('nDCG: ', nDCGs)
        print('avg nDCG: ', avgnDCG)
        return avgacc

    def MOSTPOP(self, data, mostpop, pop_list):
        # mostpop.index is the movieID
        intrctn = list(set(data.movieID).intersection(set(pop_list.index)))
        mostpop['mostPop'] = 0

        if intrctn:
            for i in intrctn[:self.topk]:
                a = mostpop.movieID[mostpop.movieID == i].index
                mostpop.loc[a, 'mostPop'] = 1
            mostpop_topk = mostpop[mostpop.mostPop == 1].copy()
            # mostpop_topk = mostpop_topk.head(self.topk)
            if mostpop_topk.shape[0] < self.topk:
                diff = self.topk - mostpop_topk.shape[0]
                mostpop_topk_ = mostpop[mostpop.mostPop == 0].copy()
                mostpop_topk_.loc[:, 'movieID'] = 0
                mostpop_topk = pd.concat([mostpop_topk, mostpop_topk_.head(diff)], axis=0)
        else:
            # index = movieID
            mostpop_topk = data.head(self.topk)
            mostpop_topk.loc[:, 'movieID'] = 0

        return mostpop_topk

    def RANDOMPOP(self, data, randompop, rndm_list):
        # mostpop.index is the movieID
        intrctn = list(set(randompop.movieID).intersection(set(rndm_list.index)))
        random.shuffle(intrctn)
        randompop['randomPop'] = 0

        if intrctn:
            for i in intrctn[:self.topk]:
                a = randompop.movieID[randompop.movieID == i].index
                randompop.loc[a, 'randomPop'] = 1
            randompop_topk = randompop[randompop.randomPop == 1].copy()
            randompop_topk = randompop_topk.head(self.topk)
            if randompop_topk.shape[0] < self.topk:
                diff = self.topk - randompop_topk.shape[0]
                randompop_topk_ = randompop[randompop.randomPop == 0].copy()
                randompop_topk_.loc[:, 'movieID'] = 0
                randompop_topk = pd.concat([randompop_topk, randompop_topk_.head(diff)], axis=0)

        else:
            # index = movieID
            randompop_topk = data.head(self.topk)
            randompop_topk.loc[:, 'movieID'] = 0
        return randompop_topk

