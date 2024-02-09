import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from classification import *
from numpy import round

class ROC_Curve(object):
    def __init__(self):
        self.ann = None
        # self.legend = ["Random classifier"]
        self.curve_list = []
        self.counter = 0


    def draw_roc_curve(self, test_y, pred_prob_y, classifier_name):
        "Plots ROC curve"

        self.counter += 1
        
        plt.title("ROC Curve")
        random_classifier = plt.plot((0, 1), (0, 1), color='black', linestyle='dashed', alpha=0.3, label="Random classifier")
        
        fpr, tpr, _ = roc_curve(test_y, pred_prob_y[:,1], pos_label=1)
        self.curve_list.append(plt.plot(fpr, tpr, label=f'{classifier_name}: {round(roc_auc_score(test_y, pred_prob_y[:,1]), 3)}'))
        
        plt.legend([random_classifier] + self.curve_list, loc='lower right') # Plotting legend
        plt.show()