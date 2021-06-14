"""
EDIT NOTICE

File edited from original in https://github.com/castorini/hedwig
by Bernal Jimenez Gutierrez (jimenezgutierrez.1@osu.edu)
in May 2020
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from document_classification.common.evaluators.evaluator import Evaluator
import pickle
import os


def get_confusion_matrix(y_true, y_pred, normalize=False, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)

    y_true_clean = []
    y_pred_clean = []
    for i in range(len(y_true)):
        if y_true[i] > 1 and y_pred[i] > 1:
            y_true_clean.append(y_true[i])
            y_pred_clean.append(y_pred[i])

    classes = np.array(['0', '1', '2', '3', '4'])
    classes = classes[unique_labels(y_true_clean, y_pred_clean)]
    cm = metrics.confusion_matrix(y_true_clean, y_pred_clean)
    print(cm)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title="Document Classification",
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
    plt.savefig('cm_document_classification.png')
    return cm


class ClassificationEvaluator(Evaluator):

    def __init__(self, dataset_cls, model, embedding, data_loader, batch_size, device, keep_results=False, args=None):
        super().__init__(dataset_cls, model, embedding, data_loader, batch_size, device, keep_results, args)
        self.ignore_lengths = False
        self.is_multilabel = False
        self.args = args

    def get_scores(self):
        self.model.eval()
        self.data_loader.init_epoch()
        total_loss = 0

        if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
            # Temporal averaging
            old_params = self.model.get_params()
            self.model.load_ema_params()

        predicted_labels, target_labels = list(), list()
        for batch_idx, batch in enumerate(self.data_loader):
            if hasattr(self.model, 'tar') and self.model.tar:
                if self.ignore_lengths:
                    scores, rnn_outs = self.model(batch.text)
                else:
                    scores, rnn_outs = self.model(batch.text[0], lengths=batch.text[1])
            else:
                if self.ignore_lengths:
                    scores = self.model(batch.text)
                else:
                    scores = self.model(batch.text[0], lengths=batch.text[1])

            if self.is_multilabel:
                scores_rounded = F.sigmoid(scores).round().long()
                predicted_labels.extend(scores_rounded.cpu().detach().numpy())
                target_labels.extend(batch.label.cpu().detach().numpy())
                total_loss += F.binary_cross_entropy_with_logits(scores, batch.label.float(), size_average=False).item()
            else:
                predicted_labels.extend(torch.argmax(scores, dim=1).cpu().detach().numpy())
                target_labels.extend(torch.argmax(batch.label, dim=1).cpu().detach().numpy())
                total_loss += F.cross_entropy(scores, torch.argmax(batch.label, dim=1), size_average=False).item()

            if hasattr(self.model, 'tar') and self.model.tar:
                # Temporal activation regularization
                total_loss += (rnn_outs[1:] - rnn_outs[:-1]).pow(2).mean()

        predicted_labels = np.array(predicted_labels)
        target_labels = np.array(target_labels)

        if self.args is not None:
            pickle.dump((predicted_labels, target_labels), open(os.path.join(self.args.data_dir,self.args.dataset,'{}_{}_{}_predictions.p'.format(self.split,self.args.model,self.args.training_file)),'wb'))

        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        precision = metrics.precision_score(target_labels, predicted_labels, average='weighted')
        recall = metrics.recall_score(target_labels, predicted_labels, average='weighted')
        f1_micro = metrics.f1_score(target_labels, predicted_labels, average='micro')
        f1_macro = metrics.f1_score(target_labels, predicted_labels, average='macro')
        f1_weighted = metrics.f1_score(target_labels, predicted_labels, average='weighted')

        avg_loss = total_loss / len(self.data_loader.dataset.examples)

        confusion_matrix = get_confusion_matrix(target_labels, predicted_labels, normalize=True)

        if self.args is not None:
            pickle.dump(([accuracy, precision, recall, f1_weighted, avg_loss], ['accuracy', 'precision', 'recall', 'f1_weighted', 'avg_loss']),open(os.path.join(self.args.data_dir,self.args.dataset,'{}_{}_{}_metrics.p'.format(self.split,self.args.model,self.args.training_file)),'wb'))

        if hasattr(self.model, 'beta_ema') and self.model.beta_ema > 0:
            # Temporal averaging
            self.model.load_params(old_params)

        return [accuracy, precision, recall, f1_micro, f1_macro, f1_weighted, avg_loss, confusion_matrix], ['accuracy', 'precision', 'recall', "f1_micro", 'f1_macro', 'f1_weighted', 'cross_entropy_loss', 'confusion_matrix']
