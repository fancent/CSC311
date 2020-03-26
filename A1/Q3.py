import numpy as np
import matplotlib.pyplot as plt


data_train = {'X': np.genfromtxt('data_train_X.csv', delimiter=','),
              't': np.genfromtxt('data_train_y.csv', delimiter=',')}
data_test = {'X': np.genfromtxt('data_test_X.csv', delimiter=','),
             't': np.genfromtxt('data_test_y.csv', delimiter=',')}

lambdaList = np.arange(0.02, 1.5, 0.03)

def shuffle_data(data):
    result = {}
    p = np.random.permutation(len(data['X']))
    result['X'], result['t'] = data['X'][p], data['t'][p]
    return result

def split_data(data, num_folds, fold):
    foldResult = {}
    restResult = {}
    restRange = [i for i in range(num_folds) if i!= fold]
    slicedX = np.asarray(np.array_split(data['X'], num_folds))
    slicedT = np.asarray(np.array_split(data['t'], num_folds))
    foldX, restX = slicedX[fold], np.concatenate(slicedX[restRange])
    foldT, restT = slicedT[fold], np.concatenate(slicedT[restRange])
    foldResult['X'], foldResult['t'] = foldX, foldT
    restResult['X'], restResult['t'] = restX, restT
    return foldResult, restResult

def train_model(data, lamd):
    xtx = np.matmul(np.transpose(data['X']), data['X'])
    inverse = np.linalg.inv(xtx + lamd*np.identity(len(data['X'][0])))
    return np.matmul(np.matmul(inverse, np.transpose(data['X'])), data['t'])

def predict(data, model):
    return data['X'].dot(model)
    #return np.asarray([np.matmul(data['X'][i],model) for i in range(len(data['X']))])

def loss(data, model):
    #xw = np.matmul(data['X'],model)
    #t = data['t']
    return np.sum((data['t'] - predict(data, model))**2)/len(data['t'])

def cross_validation(data, num_folds, lambd_seq):
    cv_error = np.zeros(len(lambd_seq))
    data = shuffle_data(data)
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0.
        for fold in range(0,num_folds):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error[i] = cv_loss_lmd / num_folds
    return cv_error

def find_errors(trainData, testData, lambd_seq):
    trainErrors = []
    testErrors = []
    for i in range(len(lambd_seq)):
        m = train_model(trainData, lambd_seq[i])
        #trainY = predict(trainData, m)
        #testY = predict(testData, m)
        trainE = loss(trainData, m)#(np.sum(trainY-trainData['t'])**2)/len(trainData['t'])
        testE = loss(testData, m)#(np.sum(testY-testData['t'])**2)/len(testData['t'])
        trainErrors.append(trainE)
        testErrors.append(testE)
    return trainErrors, testErrors

#calculating training and test error
trainErr, testErr = find_errors(data_train, data_test, lambdaList)

#calculating cv errors
fiveFoldcv = cross_validation(data_train, 5, lambdaList)
tenFoldcv = cross_validation(data_train, 10, lambdaList)

fig, graph = plt.subplots()
graph.plot(lambdaList, trainErr, "o", label="training error")
graph.plot(lambdaList, testErr, "o", label="test error")
graph.plot(lambdaList, fiveFoldcv, "o", label="5 fold cv error")
graph.plot(lambdaList, tenFoldcv, "o", label="10 fold cv error")
graph.set(xlabel='lambda range', ylabel='errors',
       title='')
graph.grid()
graph.legend()
fig.savefig("q3.png")
plt.show()