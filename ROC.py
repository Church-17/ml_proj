import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from classification import *
from numpy import round

class ROC_Curve(object):
    def __init__(self):
        pass

    def draw_roc_curve(self, test_y, pred_prob_y, classifier_name):
        "Plots ROC curve"
        
        plt.title("ROC Curve")
        _, labels = plt.gca().get_legend_handles_labels()
        if len(labels) == 0:
            plt.plot((0, 1), (0, 1), color='black', linestyle='dashed', alpha=0.3, label="Random classifier")
        
        fpr, tpr, _ = roc_curve(test_y, pred_prob_y[:,1], pos_label=1)
        plt.plot(fpr, tpr, label=f'{classifier_name}: {round(roc_auc_score(test_y, pred_prob_y[:,1]), 3)}')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='lower right') 
        plt.show()