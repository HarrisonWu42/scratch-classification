import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
import torch
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from graph_classification.utils import plot_confusion_matrix
from utils import write_json


def search(search_dic, X_train, X_test, y_train, y_test, params):
    if search_dic is not None:
        if 'max_depth' in search_dic.keys() and 'min_child_weight' in search_dic.keys():
            record = {}
            cnt = 1
            for e1 in search_dic['max_depth']:
                params['max_depth'] = e1
                for e2 in search_dic['min_child_weight']:
                    params['min_child_weight'] = e2
                    n_rounds, acc, cv_score = modelfit(X_train, X_test, y_train, y_test, params, model_name='md' + str(e1) + '_mcw' + str(e2))
                    record[cnt] = {'max_depth': e1, 'min_child_weight': e2, 'accuarcy': acc, 'cv_score(mlogloss)': cv_score}
                    cnt += 1
            write_json(record, "record1_md_mcw.json")

        if 'gamma' in search_dic.keys():
            record = {}
            cnt = 1
            for e in search_dic['gamma']:
                params['gamma'] = e
                n_rounds, acc, cv_score = modelfit(X_train, X_test, y_train, y_test, params, model_name='gamma' + str(e))
                record[cnt] = {'gamma': e, 'accuarcy': acc, 'cv_score(mlogloss)': cv_score}
                cnt += 1
            write_json(record, "record2_gamma.json")

        if 'subsample' in search_dic.keys() and 'colsample_bytree' in search_dic.keys():
            record = {}
            cnt = 1
            for e1 in search_dic['subsample']:
                params['subsample'] = e1
                for e2 in search_dic['colsample_bytree']:
                    params['colsample_bytree'] = e2
                    n_rounds, acc, cv_score = modelfit(X_train, X_test, y_train, y_test, params, model_name='sub' + str(e1) + '_cb' + str(e2))
                    record[cnt] = {'subsample': e1, 'colsample_bytree': e2, 'accuarcy': acc, 'cv_score(mlogloss)': cv_score}
                    cnt += 1
            write_json(record, "record3_sub_cb.json")

        if 'reg_alpha' in search_dic.keys() and 'reg_lambda' in search_dic.keys():
            record = {}
            cnt = 1
            for e1 in search_dic['reg_alpha']:
                params['reg_alpha'] = e1
                for e2 in search_dic['reg_lambda']:
                    params['reg_lambda'] = e2
                    n_rounds, acc, cv_score = modelfit(X_train, X_test, y_train, y_test, params, model_name='rega' + str(e1) + '_regl' + str(e2))
                    record[cnt] = {'reg_alpha': e1, 'reg_lambda': e2, 'accuarcy': acc, 'cv_score(mlogloss)': cv_score}
                    cnt += 1
            write_json(record, "record4_reg.json")

        if 'eta' in search_dic.keys():
            record = {}
            cnt = 1
            for e in search_dic['eta']:
                params['eta'] = e
                n_rounds, acc, cv_score = modelfit(X_train, X_test, y_train, y_test, params,
                                                   model_name='eta' + str(e))
                record[cnt] = {'eta': e, 'accuarcy': acc, 'cv_score(mlogloss)': cv_score}
                cnt += 1
            write_json(record, "record5_eta.json")
    else:
        n_rounds, acc, cv_score = modelfit(X_train, X_test, y_train, y_test, params, model_name="run")


def modelfit(X_train, X_test, y_train, y_test, params, model_name):
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test)
    plst = list(params.items())

    print('Start cross validation')
    res = xgb.cv(params, dtrain, num_boost_round=500, nfold=5, metrics={'mlogloss'}, seed=610, early_stopping_rounds=25)
    num_rounds = res.shape[0]
    print('The best num of trees:', num_rounds)
    cv_score = res.iloc[-1, :]['test-mlogloss-mean']
    print('CV mlogloss:', cv_score)

    plt.figure()
    res[['train-mlogloss-mean', 'test-mlogloss-mean']].plot()
    plt.savefig('./graph2vec/mlogloss' + model_name + '.jpg')

    model = xgb.train(plst, dtrain, num_rounds)  # train xgboost model
    model.save_model('./graph2vec/model/' + model_name + '.model')  # save model

    # test
    y_pred = model.predict(dtest)

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    print("accuarcy: %.6f%%" % accuracy)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print("precision:  ", precision)
    print("recall:  ", recall)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print("f1_micro: %.6f" % f1_micro)
    print("f1_macro: %.6f" % f1_macro)
    print("weighted: %.6f" % f1_weighted)

    # confusion_matrix
    class_names = np.array(['0', '1', '2', '3', '4'])
    plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Confusion matrix, without normalization')
    plt.savefig('./graph2vec/cm.jpg')
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.savefig('./graph2vec/cm_norm.jpg')

    plot_importance(model)
    plt.savefig('./graph2vec/importance' + model_name + '.jpg')
    return num_rounds, accuracy, cv_score


if __name__ == "__main__":
    print("XGBoost's version:	", xgb.__version__)

    info_path = '../../data/info.csv'
    embedding_path = '../data/embedding/embedding_64_200.pt'

    info = pd.read_csv(info_path)
    labels = info['label'].values.tolist()
    labels.reverse()
    labels = np.array(labels)

    embeddings = torch.load(embedding_path)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=123)

    params = {
        # General parameters

        # booster:
        #       gbtree(default): lifting calculation based on tree model
        #       gblinear: lifting calculation based on linear model
        'booster': 'gbtree',

        # 'nthread': 4,  # The number of threads when XGBoost is running, the default is the maximum number of threads
        # # obtained by the current system
        # 'num_feature': 4,  # Automatically set by XGBoost, no need to be set by the user

        'seed': 610,  # random number seed
        'tree_method': 'gpu_hist',  # choose gpu as a device

        # Task parameters
        'objective': 'multi:softmax',  # Multi-classification task, Objective is used to define the learning task and
        # the corresponding loss function
        'num_class': 5,  # total categories

        # Lifting parameters
        'max_depth': 6,  # default=6, maximum depth of the tree
        'min_child_weight': 1,  # default=1, the smallest sample weight sum that the leaf node continues to divide

        'gamma': 0,  # default=0, the minimum value of the loss function that needs to be reduced when the leaf nodes
        # are divided

        'subsample': 1,  # default=1, the proportion of the training model's samples to the total samples,
        # used to prevent overfitting
        'colsample_bytree': 1,  # default=1, the proportion of features sampled when building the tree

        'eta': 0.05,  # default=0.3, stepsize
    }

    start = time.time()
    search(search_dic=None, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, params=params)
    end = time.time()
    print("Running time：%.2f s" % (end - start))