import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from classification import *
from numpy import round

class ROC_Curve(object):
    def __init__(self):
        self.ann = None
        self.printed = 1
        self.aucs = []
        self.counter = 0

    def draw_roc_curve(self, test_y, pred_prob_y):
        "Plots ROC curve"
        
        plt.title("ROC Curve")
        if self.printed:
            plt.plot((0, 1), (0, 1), color='black', linestyle='dashed', alpha=0.3)
            self.printed = 0
        fpr, tpr, _ = roc_curve(test_y, pred_prob_y[:,1], pos_label=1)
        # auc = roc_auc_score(test_y, pred_prob_y[:,1])
        self.aucs.append(0)
        self.counter += 1
        plt.plot(fpr, tpr)
        if self.ann:
            self.ann.remove()
        classifier_list = list(classifier_tuple)
        self.aucs[-1] = roc_auc_score(test_y, pred_prob_y[:,1])
        for i in range(self.counter):
            classifier_list[i] = f"{classifier_list[i]}: {str(round(self.aucs[i], 3))}"
        # self.ann = plt.annotate("AUC: %.3f" % auc, (0.5, 0))
        plt.legend(['Random Classifier'] + classifier_list)
        plt.show()