import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer

n = 0
X = 0
y = 0


def getFeature(comments):
    global y, X
    xvalues = []
    xrows = []
    xcolumns = []
    ylist = []
    with open(comments, "r") as f:
        row = 0
        for line in f:
            record = [x for x in line.split(' ')]
            if int(record[0]) >= 7:
                ylist.append(1)
            elif int(record[0]) <= 4:
                ylist.append(-1)
            else:
                continue
            for item in record[1:]:
                item = item.split(':')
                wordIdx = int(item[0].strip())
                count = int(item[1].strip())
                xvalues.append(count)
                xrows.append(row)
                xcolumns.append(wordIdx)
            row += 1
    X = sparse.coo_matrix((xvalues, (xrows, xcolumns)))
    X = X.tocsr()
    #transformer = TfidfTransformer()
    #X = transformer.fit_transform(X)
    y = np.array(ylist)

try:
    file = np.load("matrix")
    X = file["X"]
    y = file["y"]
except:
    getFeature("data")
    np.savez("matrix", X=X, y=y)
