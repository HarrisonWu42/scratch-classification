import pandas as pd
from numpy import *
import time
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from graph_classification.utils import plot_confusion_matrix


def search(search_dic, X_train, X_test, y_train, y_test, params):
    if search_dic is not None:
        if 'max_delta_step' in search_dic.keys():
            record = []
            for e in search_dic['max_delta_step']:
                params['max_delta_step'] = e
                n_rounds, acc, cv_score, f1 = modelfit(X_train, X_test, y_train, y_test, params, model_name='md' + str(e))
                record.append([e, cv_score, acc, f1])
            df = pd.DataFrame(record, columns=['max_delta_step', 'cv_score', 'acc', 'f1'])
            df.to_csv("record_max_delta_step.csv", encoding='utf8', index=None)
        if 'max_depth' in search_dic.keys() and 'min_child_weight' in search_dic.keys():
            record = []
            for e1 in search_dic['max_depth']:
                params['max_depth'] = e1
                for e2 in search_dic['min_child_weight']:
                    params['min_child_weight'] = e2
                    n_rounds, acc, cv_score, f1 = modelfit(X_train, X_test, y_train, y_test, params, model_name='md' + str(e1) + '_mcw' + str(e2))
                    record.append([e1, e2, cv_score, acc, f1])
            df = pd.DataFrame(record, columns=['max_depth', 'min_child_weight', 'cv_score', 'acc', 'f1'])
            df.to_csv("record_depth_mcw.csv", encoding='utf8', index=None)
        if 'gamma' in search_dic.keys():
            record = []
            for e in search_dic['gamma']:
                params['gamma'] = e
                n_rounds, acc, cv_score, f1 = modelfit(X_train, X_test, y_train, y_test, params, model_name='gamma' + str(e))
                record.append([e, cv_score, acc, f1])
            df = pd.DataFrame(record, columns=['gamma', 'cv_score', 'acc', 'f1'])
            df.to_csv("record_gamma.csv", encoding='utf8', index=None)

        if 'subsample' in search_dic.keys() and 'colsample_bytree' in search_dic.keys():
            record = []
            for e1 in search_dic['subsample']:
                params['subsample'] = e1
                for e2 in search_dic['colsample_bytree']:
                    params['colsample_bytree'] = e2
                    n_rounds, acc, cv_score, f1 = modelfit(X_train, X_test, y_train, y_test, params, model_name='sub' + str(e1) + '_cb' + str(e2))
                    record.append([e1, e2, cv_score, acc, f1])
            df = pd.DataFrame(record, columns=['subsample', 'colsample_bytree', 'cv_score', 'acc', 'f1'])
            df.to_csv("record_sub_col.csv", encoding='utf8', index=None)

        if 'reg_alpha' in search_dic.keys():
            record = []
            for e in search_dic['reg_alpha']:
                params['reg_alpha'] = e
                n_rounds, acc, cv_score, f1 = modelfit(X_train, X_test, y_train, y_test, params, model_name='reg_alpha' + str(e))
                record.append([e, cv_score, acc, f1])
            df = pd.DataFrame(record, columns=['reg_alpha', 'cv_score', 'acc', 'f1'])
            df.to_csv("record_alpha.csv", encoding='utf8', index=None)

        if 'reg_lambda' in search_dic.keys():
            record = []
            for e in search_dic['reg_lambda']:
                params['reg_lambda'] = e
                n_rounds, acc, cv_score, f1 = modelfit(X_train, X_test, y_train, y_test, params, model_name='reg_lambda' + str(e))
                record.append([e, cv_score, acc, f1])
            df = pd.DataFrame(record, columns=['reg_lambda', 'cv_score', 'acc', 'f1'])
            df.to_csv("record_lambda.csv", encoding='utf8', index=None)

        if 'eta' in search_dic.keys():
            record = []
            for e in search_dic['eta']:
                params['eta'] = e
                n_rounds, acc, cv_score, f1 = modelfit(X_train, X_test, y_train, y_test, params, model_name='eta' + str(e))
                record.append([e, cv_score, acc, f1])
            df = pd.DataFrame(record, columns=['eta', 'cv_score', 'acc', 'f1'])
            df.to_csv("record_eta.csv", encoding='utf8', index=None)

    else:
        n_rounds, acc, cv_score, _ = modelfit(X_train, X_test, y_train, y_test, params, model_name="run")


def modelfit(X_train, X_test, y_train, y_test, params, model_name):
    dtrain = xgb.DMatrix(X_train, y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    plst = list(params.items())

    print('Start cross validation')
    res = xgb.cv(params, dtrain, num_boost_round=1000, nfold=5, metrics={'mlogloss'}, seed=610, early_stopping_rounds=25)
    num_rounds = res.shape[0]
    print('The best num of trees:', num_rounds)
    cv_score = res.iloc[-1, :]['test-mlogloss-mean']
    print('CV mlogloss:', cv_score)

    plt.figure()
    res[['train-mlogloss-mean', 'test-mlogloss-mean']].plot()
    plt.savefig('./hybird/mlogloss' + model_name + '.jpg')

    model = xgb.train(plst, dtrain, num_rounds)  # train xgboost model
    model.save_model('./hybird/model/' + model_name + '.model')  # save model

    # test
    y_pred = model.predict(dtest)

    # metrics
    accuracy = accuracy_score(y_test, y_pred)
    print("accuarcy: %.6f" % accuracy)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    print("precision: %.6f " % precision)
    print("recall:  %.6f" % recall)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print("f1 micro: %.6f" % f1_micro)
    print("f1 macro: %.6f" % f1_macro)
    print("f1 weighted: %.6f" % f1_weighted)

    # confusion_matrix
    class_names = np.array(['0', '1', '2', '3', '4'])
    plot_confusion_matrix(y_test, y_pred, classes=class_names, title='Confusion matrix, without normalization')
    plt.savefig('./hybird/cm.jpg')

    # Plot normalized confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.savefig('./hybird/cm_norm.png')

    # importance
    plot_importance(model, max_num_features=20)
    plt.savefig('./hybird/importance' + model_name + '.jpg')

    return num_rounds, accuracy, cv_score, f1_weighted


data = pd.read_csv("../data/feature.csv")
labels = data['label'].values

# Data preprocessing
data = data[['id', 'label', 'sn', 'similarity', 'teacher', 'op_amount', 'time_consuming', 'mccabe_score',
             'num_false_targets', 'num_false_targets_bk', 'num_fks', 'num_bill',
             'towards_cond', 'fks_if_condition_opcode', 'fks_if_substack_opcode']]
data = data.rename(columns={'time_consuming': 'time', 'num_false_targets': 'num_ir_roles',
                            'num_false_targets_bk': 'num_ir_roles_bk', 'num_fks': 'num_beast'})

# teacher
s = data['teacher']
ss = pd.get_dummies(s, dummy_na=False)
teacher_names = ss.columns.tolist()
teacher_renames = dict()
idx = 1
for e in teacher_names:
    teacher_renames[e] = 'teacher_' + str(idx)
    idx += 1
ss = ss.rename(columns=teacher_renames)
data = data.drop(columns=['teacher'], axis=1)
data = pd.concat([data, ss], axis=1)

# towards_cond
ss = pd.get_dummies(data['towards_cond'], dummy_na=False)
ss_columns = ss.columns.tolist()
for e in ss_columns:
    if '比尔' in e:
        data['towards_cond'].replace(e, '比尔', inplace=True)
ss = pd.get_dummies(data['towards_cond'], dummy_na=False)
towards_names = ss.columns.tolist()
towards_renames = dict()
for e in towards_names:
    towards_renames[e] = 'towards_' + e
ss = ss.rename(columns=towards_renames)
data = data.drop(columns=["towards_cond"], axis=1)
data = pd.concat([data, ss], axis=1)
data = data.rename(index=str, columns={'towards_比尔': 'towards_bill'})

# fks_if_condition_opcode
ss = pd.get_dummies(data['fks_if_condition_opcode'], dummy_na=False)
data = data.drop(columns=['fks_if_condition_opcode'], axis=1)
data = pd.concat([data, ss['sensing_touchingobject']], axis=1)
data = data.rename(index=str, columns={'sensing_touchingobject': 'condition_sensing_touching'})

# fks_if_substack_opcode
ss = pd.get_dummies(data['fks_if_substack_opcode'], dummy_na=False)
ss_columns = ss.columns.tolist()
ss_renames = dict()
for e in ss_columns:
    ss_renames[e] = 'exit_' + e
ss = ss.rename(columns=ss_renames)
data = data.drop(columns=['fks_if_substack_opcode'], axis=1)
data = pd.concat([data, ss], axis=1)


feature = data.drop(columns=['id', 'label', 'sn'], axis=1)
feature_names = feature.columns.tolist()
feature = feature.values
print(feature_names)

X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.2, random_state=610)

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

        'subsample': 0.85,  # default=1, the proportion of the training model's samples to the total samples,
        # used to prevent overfitting
        'colsample_bytree': 0.8,  # default=1, the proportion of features sampled when building the tree

        'reg_alpha': 0.1,
        'reg_lambda': 1e-5,

        'eta': 0.05,  # default=0.3, stepsize

        'max_delta_step': 0.32,  # default=0

    }

# search_params1 = {'max_delta_step': [0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4]}
# search_params2 = {
#     'max_depth': [6],
#     'min_child_weight': [0.96, 0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03, 1.04]
# }
# search_params3 = {'gamma': [i/10.0 for i in range(0, 5)]}
# search_params4 = {
#     'subsample': [0.8, 0.85, 0.9, 0.95, 1.0],
#     'colsample_bytree': [0.75, 0.8, 0.95, 0.9, 0.95, 1.0],
# }
# search_params5 = {'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]}
# search_params5 = {'reg_alpha': [0.08, 0.09, 0.1, 0.11]}
# search_params6 = {'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]}

# search_params7 = {'eta': [0.045, 0.047, 0.05, 0.052, 0.055]}

start = time.time()
search(search_dic=None, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, params=params)
end = time.time()
print("Running time：%.2f s" % (end - start))
