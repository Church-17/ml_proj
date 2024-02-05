import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from classification import compute_performances

def draw_roc_curve(test_y, pred_prob_y):
    fpr, tpr, _ = roc_curve(test_y, pred_prob_y[:,1], pos_label=1)
    auc = roc_auc_score(test_y, pred_prob_y[:,1])
    plt.plot(fpr, tpr)
    plt.annotate("AUC: %.3f" % auc, (0.8, 0))
    plt.show()