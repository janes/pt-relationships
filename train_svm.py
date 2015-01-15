__author__ = 'dsbatista'


def main():
    extractor = FeatureExtractor()
    f_train = open("train.vw", "wb")
    training(sys.argv[1], extractor, f_train)
    f_train.close()
    f_classify = open("classify.vw", "wb")
    classify(sys.argv[2], extractor, f_classify)
    f_classify.close()

"""
X = [[0], [1], [2], [3]]
>>> Y = [0, 1, 2, 3]
>>> clf = svm.SVC()
>>> clf.fit(X, Y)
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.0, kernel='rbf', max_iter=-1, probability=False, random_state=None,
shrinking=True, tol=0.001, verbose=False)
>>> dec = clf.decision_function([[1]])
>>> dec.shape[1] # 4 classes: 4*3/2 = 6
"""

if __name__ == "__main__":
    main()