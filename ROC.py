import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from classification import *
from numpy import round

class ROC_Curve(object):
    def __init__(self):
        self.ann = None
        self.printed = 1
        self.aucs = []
        self.classifier_list = []
        self.counter = 0


    def draw_roc_curve(self, test_y, pred_prob_y, classifier_name):
        "Plots ROC curve"
        
        plt.title("ROC Curve")
        if self.printed:
            plt.plot((0, 1), (0, 1), color='black', linestyle='dashed', alpha=0.3)
            self.printed = 0
        fpr, tpr, _ = roc_curve(test_y, pred_prob_y[:,1], pos_label=1)
        self.aucs.append(0)
        self.classifier_list.append(classifier_name)
        self.counter += 1
        plt.plot(fpr, tpr)
        self.aucs[-1] = roc_auc_score(test_y, pred_prob_y[:,1])
        for i in range(self.counter):   # Adding AUC for each classifier
            self.classifier_list[i] = f"{self.classifier_list[i]}: {str(round(self.aucs[i], 3))}"
        plt.legend(['Random Classifier'] + self.classifier_list, loc='lower right') # Plotting legend
        plt.show()