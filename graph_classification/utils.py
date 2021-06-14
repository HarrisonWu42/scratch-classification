import json
import shutil
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import zipfile
import networkx as nx
from tqdm import tqdm
import nltk
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def modify_file_type_dir(dir, srctype, drctype):
    """ Modify the suffix of all files in the directory
        srctype: ".xxx"
        drctype: ".xxx"
    """
    for file_name in os.listdir(dir):
        portion = os.path.splitext(file_name)
        if portion[1] == srctype:
            new_name = portion[0] + drctype
            old_name = os.path.join(dir, file_name)
            new_name = os.path.join(dir, new_name)
            os.rename(old_name, new_name)


def modify_file_type(file_path, srctype, drctype):
    """ Modify a file suffix
    srctype: ".xxx"
    drctype: ".xxx"
    """
    portion = os.path.splitext(file_path)
    if portion[1] == srctype:
        old_name = portion[0] + srctype
        new_name = portion[0] + drctype
        os.rename(old_name, new_name)


def un_zip(file_name):
    """ Unzip a single zip file """
    zip_file = zipfile.ZipFile(file_name)
    if os.path.isdir(file_name + "_files"):
        pass
    else:
        os.mkdir(file_name + "_files")
    for names in zip_file.namelist():
        zip_file.extract(names, file_name + "_files/")
    zip_file.close()


def copy_rename_file(file, new_dir_path, new_file_name):
    """ Copy a file and rename
        file:      original file
        new_dir_path: new file directory
        new_file_name:   new file name
    """
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
    new_file = os.path.join(new_dir_path, new_file_name)
    shutil.copy(file, new_file)

    return new_file


def extract_sb3_dir(srcdir, drcdir):
    """ Extract all sb3 files in a directory """
    modify_file_type(srcdir, ".sb3", ".zip")
    files = glob.glob(srcdir + "*.zip")
    for f in files:
        un_zip(f)  # unzip
        portion = os.path.splitext(f)
        dir_path = portion[0] + ".zip_files"
        file_name = portion[0].split("\\")[-1]
        new_file_name = file_name + ".json"
        copy_rename_file(dir_path + "/project.json", drcdir, new_file_name)  # Copy and rename
        shutil.rmtree(dir_path)  # Delete directory
    modify_file_type(srcdir, ".zip", ".sb3")


def extract_sb3(file, drcdir, new_id):
    """ 提取某个sb3文件到某个文件夹下
        file: file path
        drcdir: directory path, '/' as end
        new_id: a number
    """
    modify_file_type(file, ".sb3", ".zip")  # modify the suffix name
    portion = os.path.splitext(file)
    zip_file = portion[0] + ".zip"
    un_zip(zip_file)
    dir_path = portion[0] + ".zip_files"
    new_file_name = str(new_id) + ".json"
    copy_rename_file(dir_path + "/project.json", drcdir, new_file_name)  # Copy and rename
    shutil.rmtree(dir_path)  # Delete directory
    modify_file_type(zip_file, ".zip", ".sb3")


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    y_true_clean = []
    y_pred_clean = []
    for i in range(len(y_true)):
        if y_true[i] > 1 and y_pred[i] > 1:
            y_true_clean.append(y_true[i])
            y_pred_clean.append(y_pred[i])

    # Compute confusion matrix
    cm = confusion_matrix(y_true_clean, y_pred_clean)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true_clean, y_pred_clean)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title="Graph Classification",
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    y = np.append(y_true_clean, y_pred_clean)
    plt.xlim(-0.5, len(np.unique(y)) - 0.5)
    plt.ylim(len(np.unique(y)) - 0.5, -0.5)
    return ax


def plotGraphFeature(graph):
    pos = nx.shell_layout(graph)
    # nx.draw(graph, pos)
    # node_labels = nx.get_node_attributes(graph, 'name')
    # nx.draw_networkx_labels(graph, pos, labels=node_labels)
    # # edge_labels = nx.get_edge_attributes(G)
    # # nx.draw_networkx_labels(G, pos, labels=edge_labels)
    # plt.show()

    nx.draw(graph, pos)
    node_labels = nx.get_node_attributes(graph, 'opcode')
    nx.draw_networkx_labels(graph, pos, labels=node_labels)
    print(node_labels)
    # edge_labels = nx.get_edge_attributes(G)
    # nx.draw_networkx_labels(G, pos, labels=edge_labels)

    plt.show()


def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom


def extract(dict_in, dict_out):
    for key, value in dict_in.items():
        if isinstance(value, dict):  # If value itself is dictionary
            extract(value, dict_out)
        else:
            # Write to dict_out
            if key.lower() == 'name':
                dict_out.append("|||space||||")
                dict_out.append(key)
                dict_out.append(value)
                dict_out.append(",")
            elif key.lower() == 'opcode' or key.lower() == 'parent' or key.lower() == 'next' or key.lower() == 'toplevel':
                dict_out.append(key)
                dict_out.append(value)
                dict_out.append(",")
    return dict_out


def list_to_str(lis):
    a = ''
    for i in lis:
        if isinstance(i, list):
            a = a + ' ' + list_to_str(i)
        else:
            a = a + " " + str(i)
    return a


def tokenize_text(df, column):
    tokenized_pubs = []

    for text in tqdm(df[column]):
        tokenized_text = []
        sents = nltk.sent_tokenize(text)

        for sent in sents:
            sent = sent.lower()
            words = nltk.word_tokenize(sent)
            tokenized_text.append(words)

        tokenized_pubs.append(tokenized_text)

    return tokenized_pubs


def tokenize_text_bert(df, column):
    tokenized_pubs = []

    for text in tqdm(df[column]):
        tokenized_text = []
        sents = nltk.sent_tokenize(text)

        for sent in sents:
            sent = sent.lower()
            words = nltk.word_tokenize(sent)
            if words[-1] == '.':
                words = words[:-1]
            words.append('[SEP]')
            tokenized_text.append(words)

        tokenized_pubs.append(tokenized_text)

    return tokenized_pubs


def load_all_cord(source_file_path="../data/cord-19-sources"):
    files = glob.glob(source_file_path + "/clean*")

    dfs = []

    for file in files:
        print(file)
        df = pd.read_csv(file)
        df["source"] = file.split('/')[-1][:-4]

        dfs.append(df)

    covid_pubs = pd.concat(dfs)

    return covid_pubs


def load_all_litcovid(source_file_path="../data"):
    files = glob.glob(source_file_path + "/litcovid_source*.tsv")

    dfs = []

    for file in files:
        df = pd.read_csv(file, sep='\t', comment='#')
        df["source"] = file.split('/')[-1][16:-4]
        dfs.append(df)

    litcovid = pd.concat(dfs)

    return litcovid


def add_count_to_dict(term_dict, term, count=1):
    if term in term_dict:
        term_dict[term] += count
    else:
        term_dict[term] = count


def extract_ngrams_from_sent_list(sent_list, relevant_ngrams, max_length=6):
    freq_dict = {}

    for token_list in sent_list:
        num_tokens = len(token_list)
        token_list = [token.lower() for token in token_list]

        for st in range(num_tokens - 1):
            for ngram_len in range(min(max_length, num_tokens - st - 1)):
                end = st + ngram_len + 1
                ngram = ' '.join(token_list[st:end])

                if ngram in relevant_ngrams:
                    add_count_to_dict(freq_dict, ngram)

    return freq_dict


def df_from_dict(d, keys, vals):
    df = pd.DataFrame()
    df[keys] = list(d.keys())
    df[vals] = list(d.values())

    return df


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    y_true_clean = []
    y_pred_clean = []
    for i in range(len(y_true)):
        if y_true[i] > 1 and y_pred[i] > 1:
            y_true_clean.append(y_true[i])
            y_pred_clean.append(y_pred[i])

    # Compute confusion matrix
    cm = confusion_matrix(y_true_clean, y_pred_clean)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true_clean, y_pred_clean)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title="ours",
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    y = np.append(y_true_clean, y_pred_clean)
    plt.xlim(-0.5, len(np.unique(y))-0.5)
    plt.ylim(len(np.unique(y))-0.5, -0.5)
    return ax