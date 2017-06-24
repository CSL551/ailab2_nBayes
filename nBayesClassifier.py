from __future__ import division
import numpy as np
from scipy.sparse import vstack
from getFeature import X, y


def nBayesClassifier(traindata, trainlabel, testdata, testlabel, threshold):
    n = traindata.shape[0]    # number of training samples
    m = traindata.shape[1]    # size of Bag of Words
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

    # test process
    print "begin test"
    n1 = (testdata.shape)[0]    # number of test samples
    p_test = np.zeros(n1)    # postive and negtive probabilities for test samples
    for i in xrange(n1):
        p_test[i] = p_pos
        x = testdata.getrow(i)
        for wid in x.nonzero()[1]:
            wcount = x[0, wid]
            if p_wi_pos[wid] > 0:    # ensure conditional probability is nonzero
                p_test[i] *= (p_wi_pos[wid] ** wcount)
                p_test[i] /= (p_wi[wid] ** wcount)

    y_pred = np.zeros(n1)
    correct_count = 0
    for i in xrange(n1):
        if p_test[i] >= threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = -1
        if y_pred[i] == testlabel[i]:
            correct_count += 1
    return y_pred, correct_count / n1


n = (X.shape)[0]
foldsize = n // 5
with open("cross_validation.txt", "w") as f:
    for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        f.write("when threshold is {}, ".format(threshold))
        avg_accuracy = 0.0
        for i in xrange(5):
            begin = i * foldsize
            end = begin + foldsize
            traindata = vstack((X[:begin, :], X[end:, :]))
            trainlabel = np.concatenate((y[:begin], y[end:]))
            testdata = X[begin:end, :]
            testlabel = y[begin:end]
            (y_pred, accuracy) = nBayesClassifier(traindata, trainlabel, testdata, testlabel, threshold)
            avg_accuracy += accuracy
        avg_accuracy /= 5
        f.write("the average accuracy is {}.\n".format(avg_accuracy))

#nBayesClassifier(X[0 : int(0.8 * n)][:], y[0 : int(0.8 * n)], X[int(0.8 * n) : n][:], y[int(0.8 * n) : n], 0.5)