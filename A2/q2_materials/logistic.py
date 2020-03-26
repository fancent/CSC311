""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """
    z = np.dot(data, weights[:len(data[0])])
    y = sigmoid(z)
    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    #import pdb; pdb.set_trace()
    ce = np.sum(-np.multiply(targets, np.log(y)) - np.multiply((1-targets), np.log(1-y)))
    normalizedY = [[1] if i > 0.5 else [0] for i in y]
    frac_correct = np.sum(normalizedY == targets)/len(targets)
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    f , percent = evaluate(targets, y)
    dfdw = np.dot(data.T, y - targets)
    dfdb = [np.zeros(weights[-1].shape)]
    df = np.append(dfdw, dfdb, axis=0)
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """
    lambd = hyperparameters['lambd']*hyperparameters['weight_regularization']
    regulator = lambd/2 * np.sum(weights[:-1]**2)
    regulatordw = lambd/2 * np.sum(2*weights[:-1])
    y = logistic_predict(weights, data)
    f , percent = evaluate(targets, y)
    f += (regulator*hyperparameters['weight_regularization'])
    dfdw = np.dot(data.T, y - targets) + regulatordw
    dfdb = [np.zeros(weights[-1].shape)]
    df = np.append(dfdw, dfdb, axis=0)
    return f, df, y
