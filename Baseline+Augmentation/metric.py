from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from scipy.special import expit
import matplotlib.pyplot as plt
import numpy as np


class Metric:
    def __init__(self, y_true, y_pred, classes, val_classes=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.classes = classes
        self.score_list = ['precision', 'recall', 'f1_score']

        y_pred = expit(y_pred)  # Sigmoid function
        threshold = 0.5
        # self.y_pred_argmax = (y_pred >= threshold).astype(int)
        if val_classes is None:
            val_classes = classes

        self.score = {}
        f = []
        for i in range(len(val_classes)):
            cls = val_classes[i]
            cls_idx = list(classes).index(cls)
            prediction = y_pred[:, cls_idx]
            prediction = (prediction >= threshold).astype(int)
            label = y_true[:, cls_idx]
            tf = prediction == label
            true, false = prediction[tf], prediction[~tf]
            tp = len(np.where(true == 1)[0])
            tn = len(np.where(true == 0)[0])
            fp = len(np.where(false == 1)[0])
            fn = len(np.where(false == 0)[0])
            n_sample = len(np.where(label == 1)[0])

            precision, recall, f1_score = self._cal_metric(tp, fp, tn, fn)
            self.score[cls] = {}
            self.score[cls]['precision'] = precision
            self.score[cls]['recall'] = recall
            self.score[cls]['f1_score'] = f1_score
            self.score[cls]['n_sample'] = n_sample

        self.score['ovr'] = {}
        for metric_name in self.score_list:
            # computing macro-weighted average
            score = np.array([self.score[cls][metric_name] for cls in val_classes])
            num = np.array([self.score[cls]['n_sample'] for cls in val_classes])
            num = num / np.sum(num)
            # sum_val = np.sum([self.score[cls][metric_name] * self.score[cls]['n_sample'] for cls in val_classes])
            # num = np.sum([self.score[cls]['n_sample'] for cls in val_classes])
            # self.score['ovr'][metric_name] = sum_val / num
            self.score['ovr'][metric_name] = np.sum(np.multiply(score, num))

        self.score['ovr']['cpsc'] = np.mean([self.score[cls]['f1_score'] for cls in val_classes])

    @staticmethod
    def _cal_metric(tp, fp, tn, fn):
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError:
            recall = 0
        try:
            f1_score = tp / (tp + 0.5 * (fp + fn))
        except ZeroDivisionError:
            f1_score = 0
        return precision, recall, f1_score
