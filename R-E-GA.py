# -*- coding: utf-8 -*-
# Author: YeXiaona
# Date  : 2021-06-10

'''
Source code for R-E-GA
'''
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn import metrics
import numpy as np
import copy
import heapq
import csv
import data_loader_0612

###########忽略警告
import warnings

warnings.filterwarnings("ignore")

########################################################################################################################
# Parameter Setting
########################################################################################################################
iteration_size = 75
pop_size = 50

pm = 0.01
pca_split_k = 10
weak_clf_k = 10
weak_clf_size = 300
pca_part_sample_portion = 0.5
runtimes = 10
base_estimator = RandomForestClassifier(n_estimators=100, random_state=0)


########################################################################################################################
# Function
########################################################################################################################

def get_fitness_value_v9(train_X, train_y, test_X, test_y, individual, return_values=False):
    weak_clf_predicts = []
    results = []
    predict_scores = []

    now_train_X, now_train_y = train_X, train_y
    for i in range(weak_clf_k):
        clf_f1score, predict_y, need_to_strengthen, proba_score = get_weak_clf_result(now_train_X, now_train_y,
                                                                        individual[i], test_X, test_y)
        weak_clf_predicts.append(predict_y)
        results.append(clf_f1score)
        now_train_X = np.vstack((now_train_X, now_train_X[need_to_strengthen]))
        now_train_y = np.hstack((now_train_y, now_train_y[need_to_strengthen]))

        if len(predict_scores) == 0:
            predict_scores = proba_score
        else:
            predict_scores = predict_scores + proba_score

    vote_results = []
    weak_clf_predicts = np.array(weak_clf_predicts)
    for i in range(len(test_y)):
        vote_results.append(vote_v9(weak_clf_predicts[:,i]))

    f1score = f1_score(test_y, vote_results, average="macro")

    if return_values:
        acc = accuracy_score(test_y, vote_results)
        auc = roc_auc_score(test_y, (predict_scores/weak_clf_k)[:,1])
        'spe'
        confusion = metrics.confusion_matrix(test_y, vote_results)
        tn = confusion[0,0]
        fp = confusion[0,1]
        tp = confusion[1,1]
        fn = confusion[1,0]
        sen = tp / (tp + fn)
        spe = tn / (tn + fp)
        return f1score, acc, auc, results, sen, spe
    else:
        return f1score

def get_weak_clf_result(train_X, train_y, weak_clf_fs, test_X, test_y):
    estimator = base_estimator
    estimator.fit(train_X[:,weak_clf_fs],train_y)
    predict_y = estimator.predict(test_X[:,weak_clf_fs])
    train_predict_y = estimator.predict(train_X[:,weak_clf_fs])
    f1score = f1_score(test_y, predict_y, average="macro")
    proba_score = estimator.predict_proba(test_X[:,weak_clf_fs])
    need_to_strengthen = []
    for i in range(len(train_y)):
        if train_predict_y[i] != train_y[i]:
            need_to_strengthen.append(i)
    return f1score, predict_y, need_to_strengthen, proba_score

def vote_v9(ylist):
    dict = {}
    for unique_y in np.unique(ylist):
        dict[unique_y] = 0
    for y in ylist:
        dict[y] += 1
    maxvote = max(dict, key=dict.get)
    return maxvote

def output_result(validate_results, test_results):
    '''
    :param validate_results:
    :param test_results:
    :return:
    '''
    f = open('./Result/' + dataName + '.csv', 'a')
    csv_writer = csv.writer(f, lineterminator='\n')
    for i in range(runtimes):
        csv_writer.writerow(np.hstack((str(i) + " validate_results", validate_results[i])))
        csv_writer.writerow(np.hstack((str(i) + " test_results", test_results[i])))
    f.close()

def output_f1score_acc_roc(runtime, iteration_index, type,f1score,acc,roc,sen,spe):
    f = open('./Result/' + dataName + '_' + type + '.csv', 'a')
    csv_writer = csv.writer(f, lineterminator='\n')

    csv_writer.writerow(np.hstack((str(runtime) + "th " + str(iteration_index),
                                   f1score,
                                   acc,
                                   roc,
                                   sen,
                                   spe)))
    f.close()

def output_feature_selected_times(population):
    f = open('./feature_selected_times.csv', 'a')
    csv_writer = csv.writer(f, lineterminator='\n')
    csv_writer.writerow([i+1 for i in range(gene_size)])
    feature_selected_times = [0] * gene_size
    for individual in population:
        for k in range(weak_clf_k):
            for i in range(len(individual[k])):
                feature_selected_times[individual[k][i]] += 1
    csv_writer.writerow(feature_selected_times)
    f.close()



########################################################################################################################
# Main Process
########################################################################################################################
'load data'
import data_loader_0821
data = data_loader_0821.read_data(type="all")

dataNames = ["DILI_Data"]
datasets = {"DILI_Data":data}

for dataName in dataNames:
    data = datasets[dataName]
    print("==========" + dataName + "==========")
    print(data.shape)

    X, y = data[:, :-1], data[:, -1]
    y = y.astype(str)
    gene_size = X.shape[1]

    min_max_scaler = MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    for skf_time in range(1):
        skf = StratifiedKFold(n_splits=10, shuffle=True)

        runtime = 0

        validate_results = []
        test_results = []
        for train_index, test_index in skf.split(X, y):

            'k fold split data'
            train_X, train_y, test_X, test_y = X[train_index], y[train_index], X[test_index], y[test_index]
            train_X, validate_X, train_y, validate_y = train_test_split(train_X, train_y,
                                                                        test_size=0.2)

            'generate first generation'
            population = []
            for i in range(pop_size):
                individual = []
                for j in range(weak_clf_k):
                    individual.append(np.random.randint(0, gene_size, size=weak_clf_size))
                population.append(individual)

            'evaluate first generation'
            parentValues = []
            for i in range(pop_size):
                parentValues.append(get_fitness_value_v9(train_X, train_y, validate_X, validate_y, population[i],
                                                         return_values=False))

            'show original data result'
            base_estimator.fit(train_X, train_y)
            unRotated_predict_y = base_estimator.predict(test_X)
            unRotated_test_f1score = f1_score(test_y, unRotated_predict_y, average='macro')
            unRotated_predict_y = base_estimator.predict(validate_X)
            unRotated_validate_f1score = f1_score(validate_y, unRotated_predict_y, average='macro')
            unRotated_train_y = base_estimator.predict(train_X)
            unRotated_train_f1score = f1_score(train_y, unRotated_train_y, average='macro')
            print("original data result:")
            print("train:" + str(unRotated_train_f1score))
            print("validate:" + str(unRotated_validate_f1score))
            print("test:" + str(unRotated_test_f1score))

            validate_values = []
            test_values = []
            external_values = []
            'GA'
            for iteration_index in range(iteration_size):

                'crossover'
                parent_index = np.arange(0, pop_size)
                np.random.shuffle(parent_index)
                parent_left_index, parent_right_index = parent_index[0:int(pop_size / 2)], parent_index[int(pop_size / 2):]
                children = []
                for (left, right) in zip(parent_left_index, parent_right_index):
                    child_left, child_right = copy.deepcopy(population[left]), copy.deepcopy(population[right])
                    index = np.random.randint(0, weak_clf_k)    # 交换弱分类器模块
                    temp = child_left[0:index]
                    child_left[0:index] = child_right[0:index]
                    child_right[0:index] = temp
                    children.append(child_left)
                    children.append(child_right)

                'mutation'
                mutation_times = round(weak_clf_size * pm)
                for i in range(pop_size):
                    for k in range(weak_clf_k):
                        for j in range(mutation_times):
                            temp_index = np.random.randint(0, weak_clf_size)
                            children[i][k][temp_index] = np.random.randint(0, gene_size)

                'evaluate'
                childrenValues = []
                for i in range(pop_size):
                    childrenValues.append(get_fitness_value_v9(train_X, train_y, validate_X, validate_y, children[i],
                                                             return_values=False))

                'elite selection'
                result = parentValues + childrenValues
                max_num_index_list = map(result.index, heapq.nlargest(pop_size, result))
                new_population = []
                new_population_result = []
                for i in max_num_index_list:
                    if i >= pop_size:
                        new_population.append(children[i - pop_size])
                        new_population_result.append(childrenValues[i - pop_size])
                    else:
                        new_population.append(population[i])
                        new_population_result.append(parentValues[i])

                'evaluate best one'
                validate_f1score, validate_acc, validate_auc, validate_wclfresults, validate_sen, validate_spe = get_fitness_value_v9(train_X, train_y,
                                                                                    validate_X, validate_y,
                                                                                    new_population[0],
                                                                                    return_values=True)
                validate_values.append(validate_f1score)
                output_f1score_acc_roc(runtime, iteration_index, 'validate', validate_f1score, validate_acc, validate_auc, validate_sen, validate_spe)
                print(validate_wclfresults)

                test_f1score, test_acc, test_auc, test_wclfresults, test_sen, test_spe = get_fitness_value_v9(np.vstack((train_X, validate_X)),
                                                                        np.hstack((train_y, validate_y)),
                                                                        test_X, test_y,
                                                                        new_population[0],
                                                                        return_values=True)
                test_values.append(test_f1score)
                output_f1score_acc_roc(runtime, iteration_index, 'test', test_f1score, test_acc, test_auc, test_sen, test_spe)
                # output_weak_clf_results("test", test_wclfresults, test_f1score)
                print(test_wclfresults)

                'print result'
                print(
                    "skf " + str(skf_time) +
                    " run " + str(runtime) +
                    " iteration " + str(iteration_index) +
                    " : validate_result:" + str(validate_f1score) +
                    " test_result:" + str(test_f1score)
                )

                population = new_population
                parentValues = new_population_result
                print("avg:" + str(np.mean(parentValues)))

            validate_results.append(validate_values)
            test_results.append(test_values)
            runtime += 1

        output_result(validate_results, test_results)

        output_feature_selected_times(population)
