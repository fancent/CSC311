import numpy as np
import matplotlib.pyplot as plt
from check_grad import check_grad
from utils import *
from logistic import *

def run_logistic_regression():
    #train_inputs, train_targets = load_train()
    """90% valid frac with 0.1, 80, 55"""
    train_inputs, train_targets = load_train_small()
    """70$ valid frac with 0.7, 600, 500"""
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = {
                    'learning_rate': 0.002,
                    'weight_regularization': N,
                    'num_iterations': 1200,
                    'lambd': 1
                 }

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)
    
    #2.2
# =============================================================================
#     weights = 0.01 * np.random.randn(len(train_inputs[0])+1, 1)
#     tList = np.arange(0,hyperparameters['num_iterations'],1)
#     train_CEList = []
#     valid_CEList = []
#     for t in range(hyperparameters['num_iterations']):
#         # TODO: you may need to modify this loop to create plots, etc.
# 
#         # Find the negative log likelihood and its derivatives w.r.t. the weights.
#         f, df, predictions = logistic(weights, train_inputs, train_targets, hyperparameters)
#         
#         # Evaluate the prediction.
#         cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)
# 
#         if np.isnan(f) or np.isinf(f):
#             print(f)
#             raise ValueError("nan/inf error")
# 
#         # update parameters
#         weights = weights - hyperparameters['learning_rate'] * df / N
# 
#         # Make a prediction on the valid_inputs.
#         predictions_valid = logistic_predict(weights, valid_inputs)
#         # Make a prediction on test inputs
#         predictions_test = logistic_predict(weights, test_inputs)
#         
#         # Evaluate the prediction.
#         cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
#         # Evaluate the prediction for test.
#         cross_entropy_test, frac_correct_test = evaluate(test_targets, predictions_test)
#         
#         train_CEList.append(cross_entropy_train)
#         valid_CEList.append(cross_entropy_valid)
#         # print some stats
#         print(("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
#                "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f} "
#                "TEST CE:{:.6f}  TEST FRAC:{:2.2f}").format(
#                    t+1, f / N, cross_entropy_train, frac_correct_train*100,
#                    cross_entropy_valid, frac_correct_valid*100,
#                    cross_entropy_test, frac_correct_test*100))
#     fig, graph = plt.subplots()
#     graph.plot(tList, train_CEList, label="training")
#     graph.plot(tList, valid_CEList, label="valid")
#     graph.set(xlabel='number of iterations', ylabel='Cross Entropy',
#            title="Results")
#     graph.grid()
#     graph.legend()
#     fig.savefig("q2_2.png")
#     plt.show()
# =============================================================================

    #2.3
    #learning rate: 0.001, # iterations = 700
    lambdaRange = [0, 0.001, 0.01, 0.1, 1.0]
    tList = np.arange(0,hyperparameters['num_iterations'],1)
    numRerun = 5
    # Begin learning with gradient descent
    for l in lambdaRange:
        hyperparameters['lambd'] = l
        train_CEList_avg = []
        valid_CEList_avg = []
        train_CRList_avg = []
        valid_CRList_avg = []
        for i in range(numRerun):
            weights = 0.00001 * np.random.randn(len(train_inputs[0])+1, 1)
            valid_CEList = []
            train_CEList = []
            valid_CRList = []
            train_CRList = []
            for t in range(hyperparameters['num_iterations']):        
                # Find the negative log likelihood and its derivatives w.r.t. the weights.
                f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)
                
                # Evaluate the prediction.
                cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)
        
                if np.isnan(f) or np.isinf(f):
                    print(f)
                    raise ValueError("nan/inf error")
        
                # update parameters
                weights = weights - hyperparameters['learning_rate'] * df / N
        
                # Make a prediction on the valid_inputs.
                predictions_valid = logistic_predict(weights, valid_inputs)
        
                # Evaluate the prediction.
                cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
                
                train_CRList.append((1-frac_correct_train)*100)
                valid_CRList.append((1-frac_correct_valid)*100)
                train_CEList.append(f)
                valid_CEList.append(cross_entropy_valid)
                # print some stats
            print(("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} "
                   "TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}").format(
                       t+1, f / N, cross_entropy_train, frac_correct_train*100,
                       cross_entropy_valid, frac_correct_valid*100))
            valid_CEList_avg.append(valid_CEList)
            train_CEList_avg.append(train_CEList)
            valid_CRList_avg.append(valid_CRList)
            train_CRList_avg.append(train_CRList)
        train_CEList_avg = np.mean(train_CEList_avg, axis=0)
        valid_CEList_avg = np.mean(valid_CEList_avg, axis=0)
        train_CRList_avg = np.mean(train_CRList_avg, axis=0)
        valid_CRList_avg = np.mean(valid_CRList_avg, axis=0)
        fig, graph = plt.subplots()
        graph.plot(tList, train_CEList_avg, label="training")
        graph.plot(tList, valid_CEList_avg, label="valid")
        graph.set(xlabel='number of iterations', ylabel='cross entropy',
               title="Cross Entropy for Lambda = {}".format(l))
        graph.grid()
        graph.legend()
        fig.savefig("q2_3_CE_l={}.png".format(l))
        plt.show()
        fig, graph = plt.subplots()
        graph.plot(tList, train_CRList_avg, label="training")
        graph.plot(tList, valid_CRList_avg, label="valid")
        graph.set(xlabel='number of iterations', ylabel='classification rate',
               title="Classification Error for Lambda = {}".format(l))
        graph.grid()
        graph.legend()
        fig.savefig("q2_3_CR_l={}.png".format(l))
        plt.show()

def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and 
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)

if __name__ == '__main__':
    run_logistic_regression()
