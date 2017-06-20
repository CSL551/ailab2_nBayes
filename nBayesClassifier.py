from __future__ import division
import numpy as np
from scipy import sparse
from getFeature import X, y

def nBayesClassifier(traindata, trainlabel, testdata, testlabel, threshold):
    n = (traindata.shape)[0]    # number of training samples
    m = (traindata.shape)[1]    # size of Bag of Words
    p_wi_pos = np.zeros(m)    # P(wordi | positive)
    p_wi_neg = np.zeros(m)    # P(wordi | negtive)
    countpos = 0    # total number of positive training samples
    countneg = 0    # total number of negtive training samples

    # training process
    print "begin training"
    for i in xrange(n):
        x = traindata.getrow(i)
        if trainlabel[i] == 1:
            for j in x.nonzero()[1]:
                p_wi_pos[j] += x[0, j]
            countpos += 1
        else:
            for j in x.nonzero()[1]:
                p_wi_neg[j] += x[0, j]
            countneg += 1
    if countpos > 0:
        p_wi_pos /= countpos
    if countneg > 0:
        p_wi_neg /= countneg
    for i in xrange(n):
        p_wi_pos[i] = min(p_wi_pos[i], 1)
        p_wi_neg[i] = min(p_wi_neg[i], 1)
    p_pos = countpos / (countpos + countneg)
    p_neg = 1 - p_pos
    p_wi = p_wi_pos * p_pos + p_wi_neg * p_neg
    print p_wi_pos

    # test process
    print "begin test"
    n1 = (testdata.shape)[0]    # number of test samples
    p_test = np.zeros(n1)    # postive and negtive probabilities for test samples
    for i in xrange(n1):
        p_test[i] = p_pos
        x = testdata.getrow(i)
        # TODO when conditional probability is 0?
        for wid in x.nonzero()[1]:
            wcount = x[0, wid]
            if p_wi_pos[wid] > 0:
                p_test[i] *= (p_wi_pos[wid] ** wcount)
                p_test[i] /= (p_wi[wid] ** wcount)
        #p_test[i, 0] /= (p_test[i, 0] + p_test[i, 1])
        #p_test[i, 2] = 1 - p_test[i, 1]
        #p_test[i, 1] /= (p_test[i, 0] + p_test[i, 1])
    print p_test
    ypred = np.zeros(n1)
    correct_count = 0
    for i in xrange(n1):
        if p_test[i] >= threshold:
            ypred[i] = 1
        else:
            ypred[i] = -1
        if ypred[i] == testlabel[i]:
            correct_count += 1
    print correct_count / n1


n = (X.shape)[0]
nBayesClassifier(X[0 : int(0.8 * n)][:], y[0 : int(0.8 * n)], X[int(0.8 * n) : n][:], y[int(0.8 * n) : n], 0.5)