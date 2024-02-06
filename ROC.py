import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from classification import compute_performances

class ROC_Curve(object):
    def __init__(self):
        self.ann = 'ann'

    def draw_roc_curve(self, test_y, pred_prob_y):
        fpr, tpr, _ = roc_curve(test_y, pred_prob_y[:,1], pos_label=1)
        auc = roc_auc_score(test_y, pred_prob_y[:,1])
        plt.title("ROC Curve")
        plt.plot(fpr, tpr)
        plt.annotate(str(self.ann), (0.8, 0.2))
        self.ann = plt.annotate("AUC: %.3f" % auc, (0.8, 0))
        plt.show()