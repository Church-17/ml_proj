import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from classification import *
from numpy import round
from matplotlib.backend_bases import FigureCanvasBase

random_classifier_added = False

class ROC_Curve(object):
    def __init__(self):
        pass

    def draw_roc_curve(self, test_y, pred_prob_y, classifier_name):
        "Plots ROC curve"
        
        global random_classifier_added

        plt.title("ROC Curve")
        if not random_classifier_added:
            plt.plot((0, 1), (0, 1), color='black', linestyle='dashed', alpha=0.3, label="Random classifier")
            random_classifier_added = True
        
        fpr, tpr, _ = roc_curve(test_y, pred_prob_y[:,1], pos_label=1)
        plt.plot(fpr, tpr, label=f'{classifier_name}: {round(roc_auc_score(test_y, pred_prob_y[:,1]), 3)}')
        
        handles, labels = plt.gca().get_legend_handles_labels()
        bkc_rnd, bkc_lbl = handles[0], labels[0]
        handles, labels = handles[1::2], labels[1::2]
        
        plt.legend([bkc_rnd]+handles, [bkc_lbl]+labels, loc='lower right') 

        def on_close(event):
            global random_classifier_added
            random_classifier_added = False

        canvas = plt.gcf().canvas
        if isinstance(canvas, FigureCanvasBase):  # check if we're using a GUI backend
            canvas.mpl_connect('close_event', on_close)

        plt.show()
