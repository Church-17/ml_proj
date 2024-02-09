import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from classification import *
from numpy import round

class ROC_Curve(object):
    def __init__(self):
        self.ann = None
        self.curve_list = []
        self.counter = 0
        self.random_classifier_added = False

    def draw_roc_curve(self, test_y, pred_prob_y, classifier_name):
        "Plots ROC curve"

        self.counter += 1
        
        plt.title("ROC Curve")
        if not self.random_classifier_added:
            random_classifier = plt.plot((0, 1), (0, 1), color='black', linestyle='dashed', alpha=0.3, label="Random classifier")
            self.random_classifier_added = True
        else:
            plt.plot((0, 1), (0, 1), color='black', linestyle='dashed', alpha=0.3)
        
        fpr, tpr, _ = roc_curve(test_y, pred_prob_y[:,1], pos_label=1)
        curve = plt.plot(fpr, tpr, label=f'{classifier_name}: {round(roc_auc_score(test_y, pred_prob_y[:,1]), 3)}')
        self.curve_list.append(curve[0])  
        
        plt.legend(loc='lower right') 
        plt.show()

        self.__init__()
