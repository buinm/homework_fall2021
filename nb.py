import os
import sys
import numpy as np
from collections import Counter
import random
import pdb
import math
import itertools
from itertools import product
# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry,load_simulate_data, generate_q4_data
import matplotlib.pyplot as plt

# helpers to learn and traverse the tree over attributes

# pseudocounts for uniform dirichlet prior
alpha = 0.1

#--------------------------------------------------------------------------
# Naive bayes CPT and classifier
#--------------------------------------------------------------------------


class NBCPT(object):
    """
    NB Conditional Probability Table (CPT) for a child attribute.  Each child
    has only the class variable as a parent
    """

    def __init__(self, A_i):
        """
        TODO create any persistent instance variables you need that hold the
        state of the learned parameters for this CPT
            - A_i: the index of the child variable
        """
        self.A_i = A_i


        # There are two classes so there are two entries in a dictionary
        # Since the discrete state for each feature is binary
        # Let theta be probability that A_i feature takes value 1 (convenient when doing sum to compute occurrences)
        self.thetas = { 0 : 0.0, # theta_{dck}, d = A_i, c = 0, k = 1
                        1 : 0.0} # theta_{dck}, d = A_i, c = 1, k = 1

    def learn(self, A, C):
        """
        TODO
        populate any instance variables specified in __init__ to learn
        the parameters for this CPT
            - A: a 2-d numpy array where each row is a sample of assignments
            - C: a 1-d n-element numpy where the elements correspond to the
              class labels of the rows in A
        """
        assert A.shape[0] == C.shape[0]

        N = A.shape[0]
        N_1 = np.sum(C)
        N_0 = C.shape[0] - N_1

        N_ick = [0.0, 0.0]

        # Only look at column indexed at A_i of A when iterating through each row
        for i in range(N):
            clss = C[i]
            N_ick[clss] += A[i, self.A_i]
        # breakpoint()
        self.thetas[0] = (alpha + N_ick[0]) / (2 * alpha + N_0)
        self.thetas[1] = (alpha + N_ick[1]) / (2 * alpha + N_1)

    def get_cond_prob(self, entry, c):
        """ TODO
        return the conditional probability P(A_i|C=c) for the values
        specified in the example entry and class label c
            - entry: full assignment of variables
                e.g. entry = np.array([0,1,1]) means A_0 = 0, A_1 = 1, A_2 = 1
            - c: the class
        """

        if entry == 1:
            return self.thetas[c]
        return 1 - self.thetas[c]

    def is_ignored(self):
        return abs(self.thetas[0] - self.thetas[1]) < 0.1

class NBClassifier(object):
    """
    NB classifier class specification
    """
    def __init__(self, A_train, C_train=None):
        """
        TODO create any persistent instance variables you need that hold the
        state of the trained classifier and populate them with a call to
        Suggestions for the attributes in the classifier:
            - P_c: the probabilities for the class variable C
            - cpts: a list of NBCPT objects
        """

        self.features_num = A_train.shape[1]
        #thetas = np.zeros(features_num)

        self.P_c = { 0 : alpha,
                     1 : alpha}

        # A list of NBCPT objects
        self.cpts = []

        # Each class itself contains a list of parameters for each feature
        for i in range(self.features_num):
            self.cpts.append(NBCPT(i))

    def _train(self, A_train, C_train):
        """ TODO
        train your NB classifier with the specified data and class labels
        hint: learn the parameters for the required CPTs
            - A_train: a 2-d numpy array where each row is a sample of assignments
            - C_train: a 1-d n-element numpy where the elements correspond to
              the class labels of the rows in A
        """
        assert A_train.shape[0] == C_train.shape[0]
        assert A_train.shape[1] == self.features_num

        for i in range(self.features_num):
            self.cpts[i].learn(A_train, C_train)

        # Update self.P_c here
        pseudo_count = [alpha, alpha]
        pseudo_count[1] += np.sum(C_train)
        pseudo_count[0] += C_train.shape[0] - np.sum(C_train)

        self.P_c[0] = pseudo_count[0]/sum(pseudo_count)
        self.P_c[1] = pseudo_count[1]/sum(pseudo_count)

    def classify(self, entry):
        """ TODO
        return the log probabilites for class == 0 and class == 1 as a
        tuple for the given entry
        - entry: full assignment of variables
        e.g. entry = np.array([0,1,1]) means variable A_0 = 0, A_1 = 1, A_2 = 1
        NOTE this must return both the predicated label {0,1} for the class
        variable and also the log of the conditional probability of this
        assignment in a tuple, e.g. return (c_pred, logP_c_pred)
        """

        c_pred_prob = [self.P_c[0], self.P_c[1]]
        # Compute the prediction probability for each class
        for clss in [0, 1]:
            for feature, val in enumerate(entry):
                if val != -1:
                    c_pred_prob[clss] *= self.cpts[feature].get_cond_prob(val, clss)
                else:
                    # Marginalize over A_i (take into account every possible value of A_i)
                    c_pred_prob[clss] = c_pred_prob[clss] * self.cpts[feature].get_cond_prob(0, clss) \
                                        + c_pred_prob[clss] * self.cpts[feature].get_cond_prob(1, clss)

        # Normalize the values to get actual probability
        c_pred_prob /= sum(c_pred_prob)

        if c_pred_prob[0] > c_pred_prob[1]:
            return 0, math.log(c_pred_prob[0])

        return 1, math.log(c_pred_prob[1])

    def get_ignored_bills(self):
        count = 0
        for feature in range(self.features_num):
            count += self.cpts[feature].is_ignored()
        return count

    def predict_unobserved(self, entry, index):
        """ TODO
        Predicts P(A_index  | mid entry)
        Return a tuple of probabilities for A_index=0  and  A_index = 1
        We only use the 2nd value (P(A_index =1 |entry)) in this assignment
        """
        # Bayes formula, here we marginalizes over class
        #p(A=1 | entry, theta) = sum_{class} [p(entry | A=1, theta, class)*P(A | theta, class)] / [p(A=1 |) +
        #p(A=0 |)]

        p = [0.0, 0.0]
        for feature_val in [0, 1]:
            for clss in [0, 1]: # sum_{clss}
                tmp = self.cpts[index].get_cond_prob(feature_val, clss) # P(A= feature_val | theta, clss)
                for other_idx in range(16):
                    if other_idx != index:
                        tmp *= self.cpts[other_idx].get_cond_prob(entry[other_idx], clss) # p(entry | A=feature_val, theta, class)
                p[feature_val] += tmp

        p = p /sum(p)
        return p[0], p[1]

# load data
A_data, C_data = load_vote_data()


def evaluate(classifier_cls, train_subset=False, subset_size = 0):
    """
    evaluate the classifier specified by classifier_cls using 10-fold cross
    validation
    - classifier_cls: either NBClassifier or other classifiers
    - train_subset: train the classifier on a smaller subset of the training
    data
    -subset_size: the size of subset when train_subset is true
    NOTE you do *not* need to modify this function
    """
    global A_data, C_data

    A, C = A_data, C_data

    # partition train and test set for 10 rounds
    M, N = A.shape
    tot_correct = 0
    tot_test = 0
    train_correct = 0
    train_test = 0
    step = int(M / 10 + 1)
    for holdout_round, i in enumerate(range(0, M, step)):
        # print("Holdout round: %s." % (holdout_round + 1))
        A_train = np.vstack([A[0:i, :], A[i+step:, :]])
        C_train = np.hstack([C[0:i], C[i+step:]])
        A_test = A[i: i+step, :]
        C_test = C[i: i+step]
        if train_subset:
            A_train = A_train[: subset_size, :]
            C_train = C_train[: subset_size]
        # train the classifiers
        # classifier = classifier_cls._train(A_train, C_train)
        classifier_cls._train(A_train, C_train)

        train_results = get_classification_results(classifier_cls, A_train, C_train)
        # print(
        #    '  train correct {}/{}'.format(np.sum(train_results), A_train.shape[0]))
        test_results = get_classification_results(classifier_cls, A_test, C_test)
        tot_correct += sum(test_results)
        tot_test += len(test_results)
        train_correct += sum(train_results)
        train_test += len(train_results)

    return 1.*tot_correct/tot_test, 1.*train_correct/train_test

def evaluate_ignored_partisan_bills(A_data, C_data, classifier_cls, train_subset=False, subset_size = 0):
    """
    """
    A, C = A_data, C_data

    # partition train and test set for 10 rounds
    M, N = A.shape
    step = int(M / 10 + 1)
    res = []
    total_ignore_bills = 0
    train_times = 0

    for holdout_round, i in enumerate(range(0, M, step)):
        # print("Holdout round: %s." % (holdout_round + 1))
        A_train = np.vstack([A[0:i, :], A[i+step:, :]])
        C_train = np.hstack([C[0:i], C[i+step:]])
        if train_subset:
            A_train = A_train[: subset_size, :]
            C_train = C_train[: subset_size]

        # train the classifiers
        # classifier = classifier_cls._train(A_train, C_train)
        classifier_cls._train(A_train, C_train)
        total_ignore_bills += classifier_cls.get_ignored_bills()
        train_times +=1

    # Return averaged
    return 1.*total_ignore_bills/(N * train_times)

# score classifier on specified attributes, A, against provided labels,
# C
def get_classification_results(classifier, A, C):
    results = []
    pp = []
    for entry, c in zip(A, C):
        c_pred, unused = classifier.classify(entry)
        results.append((c_pred == c))
        pp.append(unused)
    # print('logprobs', np.array(pp))
    return results


def evaluate_incomplete_entry(classifier_cls):

    global A_data, C_data

    # train a classifier on the full dataset
    classifier_cls._train(A_data, C_data)

    # load incomplete entry 1
    entry = load_incomplete_entry()

    c_pred, logP_c_pred = classifier_cls.classify(entry)
    print("  P(C={}|A_observed) = {}".format(c_pred, np.exp(logP_c_pred)))

    return


def predict_unobserved(classifier_cls, index=11):
    global A_data, C_data

    # train a classifier on the full dataset
    classifier_cls._train(A_data, C_data)
    # load incomplete entry
    entry = load_incomplete_entry()

    a_pred = classifier_cls.predict_unobserved(entry, index)
    print("  P(A{}=1|A_observed) = {:2.4f}".format(index+1, a_pred[1]))

    return


def main():

    """
    TODO modify or use the following code to evaluate your implemented
    classifiers
    Suggestions on how to use the starter code for Q2, Q3, and Q5:


    ##For Q1
    print('Naive Bayes')
    accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
    print('  10-fold cross validation total test error {:2.4f} on {} '
        'examples'.format(1 - accuracy, num_examples))
    ##For Q3
    print('Naive Bayes (Small Data)')
    train_error = np.zeros(10)
    test_error = np.zeros(10)
    for x in range(10):
    accuracy, train_accuracy = evaluate(NBClassifier, train_subset=True,subset_size = (x+1)*10)
    train_error[x] = 1-train_accuracy
    test_error[x] = 1- accuracy
    print('  10-fold cross validation total test error {:2.4f} total train error {:2.4f}on {} ''examples'.format(1 - accuracy, 1- train_accuracy  ,(x+1)*10))
    print(train_error)
    print(test_error)
    ##For Q4 TODO
    ##For Q5
    print('Naive Bayes Classifier on missing data')
    evaluate_incomplete_entry(NBClassifier)

    index = 11
    print('Prediting vote of A%s using NBClassifier on missing data' % (
      index + 1))
    predict_unobserved(NBClassifier, index)
    """

    global A_data
    # Q1
    print('Q1 Naive Bayes Prediction\n')
    test_accuracy, train_accuracy = evaluate(NBClassifier(A_data), train_subset=False)
    print('  10-fold cross validation total test error {:2.4f} and total train accuracy '
          '{:2.4f} '.format(1-test_accuracy, train_accuracy))

    # Q3
    print("Q3 Naive Bayes (Small Data)\n")
    x_axis = np.linspace(10, 100, 10)
    train_error = np.zeros(10)
    test_error = np.zeros(10)
    for x in range(10):
        data_size = (x+1) * 10
        accuracy, train_accuracy = evaluate(NBClassifier(A_data), train_subset=True, subset_size=data_size)
        train_error[x] = 1 - train_accuracy
        test_error[x]  = 1 - accuracy
        print("10-fold cross validation test error {:2.4f} and train error {:2.4f} using {} data train size".format(test_error[x], train_error[x], data_size))

    # Plotting test error
    plt.plot(x_axis, test_error, 'ro')
    plt.ylabel("Test error")
    plt.xlabel("Sample size")
    #plt.show()

    # Plotting test error
    plt.plot(x_axis, train_error, 'ro')
    plt.ylabel("Train error")
    plt.xlabel("Sample size")
    #plt.show()

    print(train_error)
    print(test_error)

    # Q4
    #load synthetic data
    print("Q4 Loading data")
    simulated_data_path = './data/house-votes-simulated.complete.data'
    generate_q4_data(4000, simulated_data_path)
    A_data, C_data = load_simulate_data(simulated_data_path)


    print("Evaluating non-partisan bills")
    x_axis = np.linspace(400, 4000, 10)
    ignored_bill_rate_ratios = np.zeros(10)

    for x in range(10):
        data_size = (x+1) * 400
        ignored_bill_rate = evaluate_ignored_partisan_bills(A_data, C_data, NBClassifier(A_data), train_subset=True, subset_size=data_size)
        print("10-fold cross validation ignored bills rate {:2.4f} using {} data train size".format(ignored_bill_rate, data_size))
        ignored_bill_rate_ratios[x] = ignored_bill_rate

    # Plotting test error
    #print(x_axis)
    print(ignored_bill_rate_ratios)
    plt.plot(x_axis, ignored_bill_rate_ratios, 'ro')
    plt.ylabel("Ignored Rate")
    plt.xlabel("Sample Size")
    #plt.show()

    #For Q5
    A_data, C_data = load_vote_data()

    print('Naive Bayes Classifier on missing data')
    evaluate_incomplete_entry(NBClassifier(A_data))

    index = 11
    print('Prediting vote of A%s using NBClassifier on missing data' % (
            index + 1))
    predict_unobserved(NBClassifier(A_data), index)

if __name__ == '__main__':
    main()
