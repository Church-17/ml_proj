from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from custom_Naive_Bayes import CustomNaiveBayes
from sklearn.metrics import confusion_matrix
import numpy as np

classifier_tuple = ('Ensamble classifier', 'Decision tree', 'K nearest neighbor', 'Support Vector Classifier', 'Custom Naive Bayes')
distance_tuple = ('Uniform', 'Euclidean', 'Manhattan', 'Cosine', 'Pearson correlation')
purity_tuple = ('Gini', 'Entropy', 'LogLoss')
kernel_tuple = ('Linear', 'Polinomial', 'RBF')

def classify(classifier_str, train_x, train_y, gui_params):
    params = {}

    if classifier_str == classifier_tuple[1]:
        if gui_params['option'] == purity_tuple[0]:
            params['criterion'] = 'gini'
        elif gui_params['option'] == purity_tuple[1]:
            params['criterion'] = 'entropy'
        elif gui_params['option'] == purity_tuple[2]:
            params['criterion'] = 'log_loss'
        params['max_depth'] = None
        params['max_features'] = "sqrt"
        params['min_samples_leaf'] = 10
        params['min_samples_split'] = 10
        classifier = DecisionTreeClassifier(**params)

    elif classifier_str == classifier_tuple[2]:
        if gui_params['option'] == distance_tuple[0]:
            params['weights'] = 'uniform'
        elif gui_params['option'] == distance_tuple[1]:
            params['weights'] = 'distance'
            params['metric'] = 'euclidean'
        elif gui_params['option'] == distance_tuple[2]:
            params['weights'] = 'distance'
            params['metric'] = 'manhattan'
        elif gui_params['option'] == distance_tuple[3]:
            params['weights'] = 'distance'
            params['metric'] = 'cosine'
        elif gui_params['option'] == distance_tuple[4]:
            params['weights'] = 'distance'
            params['metric'] = 'correlation'
        params['n_neighbors'] = 10
        classifier = KNeighborsClassifier(**params)

    elif classifier_str == classifier_tuple[3]:
        if gui_params['option'] == kernel_tuple[0]:
            params['kernel'] = 'linear'
        elif gui_params['option'] == kernel_tuple[1]:
            params['kernel'] = 'poly'
        elif gui_params['option'] == kernel_tuple[2]:
            params['kernel'] = 'rbf'
        params['C'] = 10
        params['gamma'] = 10
        classifier = SVC(**params)

    elif classifier_str == classifier_tuple[4]:
        classifier = CustomNaiveBayes()
    elif classifier_str == classifier_tuple[0]:
        params['w'] = gui_params['w']
        params['voting'] = gui_params['voting']
        classifier = Custom_Ensemble(params)
        
    
    classifier.fit(train_x, train_y)

    #pred_y = classifier.predict(test_x)
    return classifier

def compute_performances(test_y, pred_y):
    cm = confusion_matrix(test_y, pred_y)
    eps = np.finfo(float).eps
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TPR = TP / (TP + FN + eps)
    TNR = TN / (TN + FP + eps)
    FPR = FP / (TN + FP + eps)
    FNR = FN / (TP + FN + eps)
    p = TP / (TP + FP + eps)
    r = TPR
    F1 = 2*r*p / (r+p + eps)
    return TPR, TNR, FPR, FNR, p, r, F1