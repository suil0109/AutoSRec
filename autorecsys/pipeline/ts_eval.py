# multiple ID eval version
import pandas as pd
import numpy as np
import random
from autorecsys.pipeline.ts_metrics import MRR_, ndcg_
# from sklearn.metrics import ndcg_score, dcg_score
from sklearn.utils import shuffle


class Eval_Topk():
    def     __init__(self, pred, data, topk, timestep=None):
        # pred has ratings only
        # data has all x and ys
        self.pred = pred
        self.data = data
        self.topk = topk
        self.timestep = timestep

    def HR_MRR_nDCG(self):
        pred = pd.DataFrame(self.pred.flatten(), columns=['pred'])
        data = self.data[['rating', 'userID', 'movieID']]

        # disregard the first timestep
        data = data.iloc[self.timestep:]
        data = data.reset_index(drop=True)
        combined = pd.concat([pred, data], axis=1)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)
        # print(combined)

        list_a = combined['userID'].unique()
        # get most interactions
        pop_list = combined.movieID.value_counts().head(200)
        rndm_list = combined.movieID.value_counts().head(200)
        accuracy = []
        mrr = []
        nDCGs = []
        accuracy1 = []
        mrr1 = []
        nDCGs1 = []
        accuracy2 = []
        mrr2 = []
        nDCGs2 = []
        length = []

        # important: reverse the order
        # order from old -> new to new -> old, so that pd sort the newest first
        # tmp = combined[::-1]
        tmp = combined.groupby(["userID"])


        y = []
        for x in list_a:
            y.append(x)
            tmp_ = tmp.get_group(x)
            length.append(tmp_.shape[0])
            pred = tmp_.sort_values(by=['pred'], ascending=False)
            # cat_val = pred[['Action', 'Adventure',
            #                            'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
            #                            'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
            #                            'Sci-Fi', 'Thriller', 'War', 'Western']].sum()
            # cat_val = cat_val.sort_values(ascending=False)
            true = tmp_.copy()

            # data = tmp_.sort_values(by=['rating'], ascending=False)
            # mostpop.index is the movieID
            mostpop_topk = self.MOSTPOP(data.copy(), tmp_.copy(), pop_list)
            random_topk = self.RANDOMPOP(data.copy(), tmp_.copy(), rndm_list)
            # print(random_topk)
            random_topk = random_topk.sample(frac=1)
            # print(random_topk)
            true_topk = true.head(self.topk)
            pred_topk = pred.head(self.topk)
            # pred_topk = pred_topk.sort_values(by=['releaseDate'], ascending=False)
            # pred_topk = pred_topk.sort_values(by=['releaseDate'], ascending=True)

            # pd.set_option('display.max_columns', None)
            # pd.set_optionp('display.max_rows', None)

            # Random2
            # Accuracy
            accuracy.append(len(np.intersect1d(true_topk.movieID.values, random_topk.movieID.values)) / len(true_topk))

            # MRR
            ovrlp = np.intersect1d(true_topk.movieID.values, random_topk.movieID.values)
            ovrlp = list(ovrlp)
            mrr_ = MRR_(true_topk.copy(), ovrlp)
            mrr.append(mrr_)

            # nDCG
            nDCG_ = ndcg_(true_topk.copy(), random_topk.copy(), ovrlp)
            nDCGs.append(nDCG_)

            # Most_Pop
            accuracy1.append(len(np.intersect1d(true_topk.movieID.values, mostpop_topk.movieID.values)) / len(true_topk))

            # MRR
            ovrlp = np.intersect1d(true_topk.movieID.values, mostpop_topk.movieID.values)
            ovrlp = list(ovrlp)
            mrr_ = MRR_(true_topk.copy(), ovrlp)
            mrr1.append(mrr_)

            # nDCG
            nDCG_ = ndcg_(true_topk.copy(), mostpop_topk.copy(), ovrlp)
            nDCGs1.append(nDCG_)

            # Pred
            accuracy2.append(len(np.intersect1d(true_topk.movieID.values, pred_topk.movieID.values)) / len(true_topk))
            bb = len(np.intersect1d(true_topk.movieID.values, pred_topk.movieID.values)) / len(true_topk)
            if bb > 0.4:
                print(len(np.intersect1d(true_topk.movieID.values, pred_topk.movieID.values)) / len(true_topk), '+++++')
                print(true_topk)
                print('\nhh')
                print(pred_topk)
            # MRR
            ovrlp = np.intersect1d(true_topk.movieID.values, pred_topk.movieID.values)
            ovrlp = list(ovrlp)
            mrr_ = MRR_(true_topk.copy(), ovrlp)
            mrr2.append(mrr_)

            # nDCG
            nDCG_ = ndcg_(true_topk.copy(), pred_topk.copy(), ovrlp)
            nDCGs2.append(nDCG_)


        # Accuracy
        print('---------------------random---------------------')
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

        print('---------------------mostpop---------------------')
        avgacc = round((sum(accuracy1) / len(accuracy1)), 4)
        print('accuracy: ', accuracy1)
        print('avg accuracy: ', avgacc)

        # MRR
        avgmrr = round((sum(mrr1) / len(mrr1)), 4)
        print('mrr: ', mrr1)
        print('avg mrr: ', avgmrr)

        # nDCG
        print('mynDCG')
        avgnDCG = round((sum(nDCGs1) / len(nDCGs1)), 4)
        print('nDCG: ', nDCGs1)
        print('avg nDCG: ', avgnDCG)

        # Accuracy
        print('---------------------pred---------------------')
        avgacc = round((sum(accuracy2) / len(accuracy2)), 4)
        print('accuracy: ', accuracy2)
        print('avg accuracy: ', avgacc)

        # MRR
        avgmrr = round((sum(mrr2) / len(mrr2)), 4)
        print('mrr: ', mrr2)
        print('avg mrr: ', avgmrr)

        # nDCG
        avgnDCG = round((sum(nDCGs2) / len(nDCGs2)), 4)
        print('nDCG: ', nDCGs2)
        print('avg nDCG: ', avgnDCG)

        # print('length: ', list(map(list,zip(accuracy2,y))))

        return avgacc

    def MOSTPOP(self, data, mostpop, pop_list):
        # mostpop.index is the movieID
        intrctn = list(set(mostpop.movieID).intersection(set(pop_list.index)))
        mostpop['mostPop'] = 0
        if intrctn:
            for i in intrctn[:self.topk]:
                a = mostpop.movieID[mostpop.movieID == i].index
                mostpop.loc[a, 'mostPop'] = 1
            mostpop_topk = mostpop[mostpop.mostPop == 1].copy()
            mostpop_topk = mostpop_topk.head(self.topk)

            if mostpop_topk.shape[0] < self.topk:
                diff = self.topk - mostpop_topk.shape[0]
                mostpop_topk_ = mostpop[mostpop.mostPop == 0].copy()
                mostpop_topk_.loc[:, 'movieID'] = 0
                mostpop_topk = pd.concat([mostpop_topk, mostpop_topk_.head(diff)], axis=0)
                shuffle(mostpop_topk)
        else:
            # index = movieID
            mostpop_topk = data.head(self.topk).copy()
            mostpop_topk.loc[:, 'movieID'] = 0
            shuffle(mostpop_topk)
        return mostpop_topk

    def RANDOMPOP(self, data, randompop, rndm_list):
        # mostpop.index is the movieID
        # print('in RANDOMPOP')
        intrctn = list(set(randompop.movieID).intersection(set(rndm_list.index)))
        # print(randompop.movieID)
        # print(rndm_list.index)
        # print(intrctn)
        random.shuffle(intrctn)
        # print('after shuffle')
        # print(intrctn)
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
            randompop_topk = data.head(self.topk).copy()
            randompop_topk.loc[:, 'movieID'] = 0
        return randompop_topk

















# next item eval


class NextItem_Eval_Topk():
    def __init__(self, pred, data, topk, timestep=None, position=None, check=None):
        # pred has ratings only
        # data has all x and ys
        self.pred = pred
        self.data = data
        self.topk = topk
        self.timestep = timestep
        self.position = position
        self.check = check

    def HR_MRR_nDCG(self):
        pred = pd.DataFrame(self.pred.flatten(), columns=['pred'])
        data = self.data[['rating', 'userID', 'movieID']]

        # disregard the first timestep
        data = data.iloc[self.timestep:]
        data = data.reset_index(drop=True)
        combined = pd.concat([pred, data], axis=1)
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_rows', None)
        # print(combined)

        list_a = combined['userID'].unique()
        # get most interactions
        pop_list = combined.movieID.value_counts().head(100)
        rndm_list = combined.movieID.value_counts().head(200)
        accuracy = []
        mrr = []
        nDCGs = []
        accuracy1 = []
        mrr1 = []
        nDCGs1 = []
        accuracy2 = []
        mrr2 = []
        nDCGs2 = []
        length = []

        # important: reverse the order
        # order from old -> new to new -> old, so that pd sort the newest first
        # tmp = combined[::-1]
        tmp = combined.groupby(["userID"])
        acc = 0
        accm = 0
        accr = 0



        y = []
        num=0
        total=0
        for x in list_a:
            y.append(x)
            tmp_ = tmp.get_group(x)
            length.append(tmp_.shape[0])
            pred = tmp_.sort_values(by=['pred'], ascending=False)
            # cat_val = pred[['Action', 'Adventure',
            #                            'Animation', "Children's", 'Comedy', "Crime", 'Documentary', 'Drama',
            #                            'Fantasy', "Film-Noir", 'Horror', 'Musical', 'Mystery', 'Romance',
            #                            'Sci-Fi', 'Thriller', 'War', 'Western']].sum()
            # cat_val = cat_val.sort_values(ascending=False)
            true = tmp_.copy()

            # data = tmp_.sort_values(by=['rating'], ascending=False)
            # mostpop.index is the movieID
            mostpop_topk = self.MOSTPOP(data.copy(), tmp_.copy(), pop_list)
            random_topk = self.RANDOMPOP(data.copy(), tmp_.copy(), rndm_list)
            # print(random_topk)
            random_topk = random_topk.sample(frac=1)
            # print(random_topk)
            true_topk = true.iloc[self.position[num]-self.timestep, :]
            pred_topk = pred.head(self.topk)

            a = np.array(true_topk.movieID)
            b = np.array(pred_topk.movieID)
            print(a, b)

            c = np.intersect1d(a, b)

            if c.shape[0] != 0:
                acc+=1

            mostpop = np.intersect1d(a, mostpop_topk.movieID)
            if mostpop.shape[0] != 0:
                accm+=1

            random_topk = np.intersect1d(a, random_topk.movieID)
            if random_topk.shape[0] != 0:
                accr+=1
            num+=1


            # intrsctn = set(true_topk.movieID).intersection(set(pred_topk.movieID))
            # print(intrsctn)

            # print(true_topk)
            # print('true_topk')
            # print(true_topk)
            # print('pred_topk')
            # print(random_topk)

            # pred_topk = pred_topk.sort_values(by=['releaseDate'], ascending=False)
            # pred_topk = pred_topk.sort_values(by=['releaseDate'], ascending=True)

            # pd.set_option('display.max_columns', None)
            # pd.set_optionp('display.max_rows', None)

            # Random2
            # Accuracy
            # accuracy.append(len(np.intersect1d(true_topk.movieID, random_topk.movieID.values)) / len(true_topk))

            # # MRR
            # ovrlp = np.intersect1d(true_topk.movieID.values, random_topk.movieID.values)
            # ovrlp = list(ovrlp)
            # mrr_ = MRR_(true_topk.copy(), ovrlp)
            # mrr.append(mrr_)
            #
            # # nDCG
            # nDCG_ = ndcg_(true_topk.copy(), random_topk.copy(), ovrlp)
            # nDCGs.append(nDCG_)
            #
            # Most_Pop
            # accuracy1.append(len(np.intersect1d(true_topk.movieID, mostpop_topk.movieID.values)) / len(true_topk))

            # # MRR
            # ovrlp = np.intersect1d(true_topk.movieID.values, mostpop_topk.movieID.values)
            # ovrlp = list(ovrlp)
            # mrr_ = MRR_(true_topk.copy(), ovrlp)
            # mrr1.append(mrr_)
            #
            # # nDCG
            # nDCG_ = ndcg_(true_topk.copy(), mostpop_topk.copy(), ovrlp)
            # nDCGs1.append(nDCG_)

            # Pred
            # accuracy2.append(len(np.intersect1d(true_topk.movieID, pred_topk.movieID.values)) / len(true_topk))
            # bb = len(np.intersect1d(true_topk.movieID, pred_topk.movieID.values)) / len(true_topk)
            # if bb > 0.2:
            #     print(len(np.intersect1d(true_topk.movieID, pred_topk.movieID.values)) / len(true_topk), '+++++')
            #     print(true_topk)
            #     print('\nhh')
            #     print(pred_topk)
            # # MRR
            # ovrlp = np.intersect1d(true_topk.movieID.values, pred_topk.movieID.values)
            # ovrlp = list(ovrlp)
            # mrr_ = MRR_(true_topk.copy(), ovrlp)
            # mrr2.append(mrr_)
            #
            # # nDCG
            # nDCG_ = ndcg_(true_topk.copy(), pred_topk.copy(), ovrlp)
            # nDCGs2.append(nDCG_)

        print('most pop accuracy:', accm/num)
        print('random accuracy:', accr/num)
        print('pred accuracy:', acc/num)


        # Accuracy
        # print('---------------------random---------------------')
        # avgacc = round((sum(accuracy) / len(accuracy)), 4)
        # print('accuracy: ', accuracy)
        # print('avg accuracy: ', avgacc)

        # # MRR
        # avgmrr = round((sum(mrr) / len(mrr)), 4)
        # print('mrr: ', mrr)
        # print('avg mrr: ', avgmrr)
        #
        # # nDCG
        # avgnDCG = round((sum(nDCGs) / len(nDCGs)), 4)
        # print('nDCG: ', nDCGs)
        # print('avg nDCG: ', avgnDCG)
        #
        # print('---------------------mostpop---------------------')
        # avgacc = round((sum(accuracy1) / len(accuracy1)), 4)
        # print('accuracy: ', accuracy1)
        # print('avg accuracy: ', avgacc)

        # # MRR
        # avgmrr = round((sum(mrr1) / len(mrr1)), 4)
        # print('mrr: ', mrr1)
        # print('avg mrr: ', avgmrr)
        #
        # # nDCG
        # print('mynDCG')
        # avgnDCG = round((sum(nDCGs1) / len(nDCGs1)), 4)
        # print('nDCG: ', nDCGs1)
        # print('avg nDCG: ', avgnDCG)

        # Accuracy
        # print('---------------------pred---------------------')
        # avgacc = round((sum(accuracy2) / len(accuracy2)), 4)
        # print('accuracy: ', accuracy2)
        # print('avg accuracy: ', avgacc)

        # # MRR
        # avgmrr = round((sum(mrr2) / len(mrr2)), 4)
        # print('mrr: ', mrr2)
        # print('avg mrr: ', avgmrr)
        #
        # # nDCG
        # avgnDCG = round((sum(nDCGs2) / len(nDCGs2)), 4)
        # print('nDCG: ', nDCGs2)
        # print('avg nDCG: ', avgnDCG)

        # print('length: ', list(map(list,zip(accuracy2,y))))

    def MOSTPOP(self, data, mostpop, pop_list):
        # mostpop.index is the movieID
        intrctn = list(set(mostpop.movieID).intersection(set(pop_list.index)))
        mostpop['mostPop'] = 0
        if intrctn:
            for i in intrctn[:self.topk]:
                a = mostpop.movieID[mostpop.movieID == i].index
                mostpop.loc[a, 'mostPop'] = 1
            mostpop_topk = mostpop[mostpop.mostPop == 1].copy()
            mostpop_topk = mostpop_topk.head(self.topk)

            if mostpop_topk.shape[0] < self.topk:
                diff = self.topk - mostpop_topk.shape[0]
                mostpop_topk_ = mostpop[mostpop.mostPop == 0].copy()
                mostpop_topk_.loc[:, 'movieID'] = 0
                mostpop_topk = pd.concat([mostpop_topk, mostpop_topk_.head(diff)], axis=0)
                shuffle(mostpop_topk)
        else:
            # index = movieID
            mostpop_topk = data.head(self.topk).copy()
            mostpop_topk.loc[:, 'movieID'] = 0
            shuffle(mostpop_topk)
        return mostpop_topk

    def RANDOMPOP(self, data, randompop, rndm_list):
        # mostpop.index is the movieID
        # print('in RANDOMPOP')
        intrctn = list(set(randompop.movieID).intersection(set(rndm_list.index)))
        # print(randompop.movieID)
        # print(rndm_list.index)
        # print(intrctn)
        random.shuffle(intrctn)
        # print('after shuffle')
        # print(intrctn)
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
            randompop_topk = data.head(self.topk).copy()
            randompop_topk.loc[:, 'movieID'] = 0
        return randompop_topk

