from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from IPython.display import Image 
import numpy as np
import pydot
import math

# =============================================================================
# a = CountVectorizer()
# cleanFake = open('clean_fake.txt')
# result = [line for line in cleanFake.readlines()]
# b = a.fit_transform(result)
# =============================================================================

def load_data(realFileName, fakeFileName):
    vect = CountVectorizer()
    realFile = open(realFileName)
    fakeFile = open(fakeFileName)
    realData = [line for line in realFile.readlines()]
    fakeData = [line for line in fakeFile.readlines()]
    realDataValue = [1 for i in range(len(realData))]
    fakeDataValue = [0 for i in range(len(fakeData))]
    combinedData = realData + fakeData
    combinedValue = realDataValue + fakeDataValue
    transformedData = vect.fit_transform(combinedData)
    trainData, restData, trainValue, restValue = train_test_split(transformedData.toarray(), combinedValue, test_size=0.3, train_size=0.7)
    validData, testData = restData[:len(restData)//2], restData[len(restData)//2:]
    validValue, testValue = restValue[:len(restValue)//2], restValue[len(restValue)//2:]
    featureNames = vect.get_feature_names()
    return trainData, validData, testData, trainValue, validValue, testValue, featureNames

# =============================================================================
# giniTree = DecisionTreeClassifier(criterion = "entropy", max_depth=30)
# giniTree.fit(xtrain, ytrain)
# gpred = giniTree.predict(xvalid)
# test = np.asarray([1 if i!=0 else 0 for i in (gpred-yvalid)])
# testacc = 1 - np.sum(test)/len(test)
# print("accuracy", accuracy_score(yvalid, gpred)*100)
# print("my acc", testacc * 100)
# =============================================================================

def select_model(depthList, xTrain, yTrain, xValid, yValid):
    for l in depthList:
        giniTree = DecisionTreeClassifier(criterion = "gini", max_depth=l)
        infoTree = DecisionTreeClassifier(criterion = "entropy", max_depth=l)
        giniTree.fit(xTrain, yTrain)
        infoTree.fit(xTrain, yTrain)
        giniPrediction = giniTree.predict(xValid)
        infoPrediction = infoTree.predict(xValid)
        giniAccuracy = np.asarray([1 if i!=0 else 0 for i in (giniPrediction-yValid)])
        infoAccuracy = np.asarray([1 if i!=0 else 0 for i in (infoPrediction-yValid)])
        giniScore = 1 - np.sum(giniAccuracy)/len(giniAccuracy)
        infoScore = 1 - np.sum(infoAccuracy)/len(infoAccuracy)
        print("accuracy for gini index with max_depth={}:".format(l), giniScore*100)
        print("accuracy for information gain with max_depth={}:".format(l), infoScore*100)
        

xtrain1, xvalid1, xtest1, ytrain1, yvalid1, ytest1, names = load_data("clean_real.txt","clean_fake.txt")
select_model([6,12,21,30,45], xtrain1, ytrain1, xvalid1, yvalid1)
bestTree = DecisionTreeClassifier(criterion = "entropy", max_depth=45)
bestTree.fit(xtrain1, ytrain1)

export_graphviz(bestTree,out_file="tree.dot",feature_names=names, max_depth=2)
(graph,) = pydot.graph_from_dot_file("tree.dot")
graph.write_png('tree.png')

# =============================================================================
# left_orange = []
# left_lemon = []
# for i in range(len(xtrain1)):
#     if xtrain1[:,names.index('the')][i] <= 0.5:
#         if ytrain1[i] == 1:
#             left_orange.append(xtrain1[i,:])
#         else:
#             left_lemon.append(xtrain1[i,:])
# =============================================================================

def compute_information_gain(Y_train, xi, X_train, featureMatrix):
    left_orange = []
    left_lemon = []
    right_orange = []
    right_lemon =[]
    for i in range(len(X_train)):
        if X_train[:,featureMatrix.index(xi)][i] <= 0.5:
            if Y_train[i] == 1:
                left_orange.append(X_train[i,:])
            else:
                left_lemon.append(X_train[i,:])
        else:
            if Y_train[i] == 1:
                right_orange.append(X_train[i,:])
            else:
                right_lemon.append(X_train[i,:])
    hy = -(911/2286)*math.log(911/2286,2) - (1375/2286)*math.log(1375/2286,2)
    left_len = len(left_orange)+len(left_lemon)
    right_len = len(right_orange)+len(right_lemon)
    hyleft = (left_len/2286)*(-(len(left_orange)/left_len)*math.log(len(left_orange)/left_len, 2)-(len(left_lemon)/left_len)*math.log(len(left_lemon)/left_len, 2))
    hyright = (right_len/2286)*(-(len(right_orange)/right_len)*math.log(len(right_orange)/right_len, 2)-(len(right_lemon)/right_len)*math.log(len(right_lemon)/right_len, 2))
    return hy - (hyleft + hyright)

print()
print("information gain:",compute_information_gain(ytrain1, 'donald',xtrain1, names))
