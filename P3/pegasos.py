import json
import numpy as np


###### Q1.1 ######
def objective_function(X, y, w, lamb):
    """
    Inputs:
    - Xtrain: A 2 dimensional numpy array of data (number of samples x number of features)
    - ytrain: A 1 dimensional numpy array of labels (length = number of samples )
    - w: a numpy array of D elements as a D-dimension weight vector
    - lamb: lambda used in pegasos algorithm

    Return:
    - obj_value: the value of objective function in SVM primal formulation
    """
    # you need to fill in your solution here

    WW = np.dot(X, w)
    SUM = 0
    for i in range(len(WW)):
        te = 1-y[i]*WW[i]
        if te > 0:
            SUM += te
        else:
            SUM += 0

    avgsum = SUM*1.0/len(WW)

    WS = 0
    for i in range(len(w)):
        WS = WS + np.power(w[i, ], 2)

    returnlist = WS * 1.0 * lamb / 2 + avgsum

    obj_value = returnlist[0]
    return obj_value


###### Q1.2 ######
def pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations):
    """
    Inputs:
    - Xtrain: A list of num_train elements, where each element is a list of D-dimensional features.
    - ytrain: A list of num_train labels
    - w: a numpy array of D elements as a D-dimension vector, which is the weight vector and initialized to be all 0s
    - lamb: lambda used in pegasos algorithm
    - k: mini-batch size
    - max_iterations: the total number of iterations to update parameters

    Returns:
    - learnt w
    - train_obj: a list of the objective function value at each iteration during the training process, length of 500.
    """
    np.random.seed(0)
    Xtrain = np.array(Xtrain)
    ytrain = np.array(ytrain)
    N = Xtrain.shape[0]
    D = Xtrain.shape[1]

    train_obj = []

    for iter in range(1, max_iterations + 1):
        A_t = np.floor(np.random.rand(k) * N).astype(int)  # index of the current mini-batch
        A_tp = []
        for i in range(len(A_t)):
            if ytrain[A_t[i]] * np.dot(Xtrain[A_t[i]], w) < 1:
                A_tp.append(A_t[i])

        Sit = 1.0 / (iter*lamb)

        TW = np.zeros(D)
        for i in range(len(A_tp)):
            TW = TW + Xtrain[A_tp[i]] * ytrain[A_tp[i]]

        TW = TW * Sit / k

        Whalf = (1 - Sit*lamb) * w + np.transpose(TW).reshape(D, 1)

        TWhalf = 0
        for i in range(len(Whalf)):
            TWhalf = TWhalf + np.power(Whalf[i, ], 2)

        TWhalf = np.sqrt(TWhalf)

        TPP = 1.0 / np.sqrt(lamb) / TWhalf

        if TPP > 1:
            w = Whalf
        else:
            w = TPP * Whalf

        train_obj.append(objective_function(Xtrain, ytrain, w, lamb))


        # you need to fill in your solution here



    return w, train_obj


###### Q1.3 ######
def pegasos_test(Xtest, ytest, w_l):
    """
    Inputs:
    - Xtest: A list of num_test elements, where each element is a list of D-dimensional features.
    - ytest: A list of num_test labels
    - w_l: a numpy array of D elements as a D-dimension vector, which is the weight vector of SVM classifier and learned by pegasos_train()
 
    Returns:
    - test_acc: testing accuracy.
    """
    # you need to fill in your solution here

    AC = 0
    yG = np.dot(Xtest, w_l)
    for i in range(len(yG)):
        if yG[i, ] >= 0:
            yG[i, ] = 1
        else:
            yG[i, ] = -1

        if yG[i, ] == ytest[i]:
            AC = AC + 1

    test_acc = AC * 1.0 / len(yG)
    		


    return test_acc


"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

def data_loader_mnist(dataset):

    with open(dataset, 'r') as f:
            data_set = json.load(f)
    train_set, valid_set, test_set = data_set['train'], data_set['valid'], data_set['test']

    Xtrain = train_set[0]
    ytrain = train_set[1]
    Xvalid = valid_set[0]
    yvalid = valid_set[1]
    Xtest = test_set[0]
    ytest = test_set[1]

    ## below we add 'one' to the feature of each sample, such that we include the bias term into parameter w
    Xtrain = np.hstack((np.ones((len(Xtrain), 1)), np.array(Xtrain))).tolist()
    Xvalid = np.hstack((np.ones((len(Xvalid), 1)), np.array(Xvalid))).tolist()
    Xtest = np.hstack((np.ones((len(Xtest), 1)), np.array(Xtest))).tolist()

    for i, v in enumerate(ytrain):
        if v < 5:
            ytrain[i] = -1.
        else:
            ytrain[i] = 1.
    for i, v in enumerate(ytest):
        if v < 5:
            ytest[i] = -1.
        else:
            ytest[i] = 1.

    return Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest


def pegasos_mnist():

    test_acc = {}
    train_obj = {}

    Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest = data_loader_mnist(dataset = 'mnist_subset.json')

    max_iterations = 500
    k = 100
    for lamb in (0.01, 0.1, 1):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    lamb = 0.1
    for k in (1, 10, 1000):
        w = np.zeros((len(Xtrain[0]), 1))
        w_l, train_obj['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_train(Xtrain, ytrain, w, lamb, k, max_iterations)
        test_acc['k=' + str(k) + '_lambda=' + str(lamb)] = pegasos_test(Xtest, ytest, w_l)

    return test_acc, train_obj


def main():
    test_acc, train_obj = pegasos_mnist() # results on mnist
    print('mnist test acc \n')
    for key, value in test_acc.items():
        print('%s: test acc = %.4f \n' % (key, value))

    with open('pegasos.json', 'w') as f_json:
        json.dump([test_acc, train_obj], f_json)


if __name__ == "__main__":
    main()
