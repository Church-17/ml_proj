from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from custom_Naive_Bayes import CustomNaiveBayes
from cursom_Ensamble import Custom_Ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np

classifier_tuple = ('Decision tree', 'K nearest neighbor', 'Support Vector Classifier', 'Custom Naive Bayes', 'Ensamble custom')
distance_tuple = ('Uniform', 'Euclidean', 'Manhattan', 'Cosine', 'Pearson correlation')
purity_tuple = ('Gini', 'Entropy', 'LogLoss')
kernel_tuple = ('Linear', 'Polinomial', 'RBF')

def training(classifier_str, train_x, train_y, gui_params):
    params = {}

    if classifier_str == classifier_tuple[0]:
        if gui_params['option'] == purity_tuple[0]:
            params['criterion'] = 'gini'
        elif gui_params['option'] == purity_tuple[1]:
            params['criterion'] = 'entropy'
        elif gui_params['option'] == purity_tuple[2]:
            params['criterion'] = 'log_loss'

        classifier = DecisionTreeClassifier()

        if gui_params['tuning']:
            params['max_depth'] = None + tuple(range(2, 30))
            params['max_features'] = ('sqrt', 'log2')
            params['min_samples_leaf'] = tuple(range(1, 15))
            params['min_samples_split'] = tuple(range(2, 15))
        else:
            params['max_depth'] = None
            params['max_features'] = "sqrt"
            params['min_samples_leaf'] = 10
            params['min_samples_split'] = 10

    elif classifier_str == classifier_tuple[1]:
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

        classifier = KNeighborsClassifier()

        if gui_params['tuning']:
            params['n_neighbors'] = tuple(range(1, 50))
        else:
            params['n_neighbors'] = 10

    elif classifier_str == classifier_tuple[2]:
        if gui_params['option'] == kernel_tuple[0]:
            params['kernel'] = 'linear'
        elif gui_params['option'] == kernel_tuple[1]:
            params['kernel'] = 'poly'
        elif gui_params['option'] == kernel_tuple[2]:
            params['kernel'] = 'rbf'

        classifier = SVC()
        
        if gui_params['tuning']:
            params['C'] = tuple(range(0.1, 10, 0.1))
            params['gamma'] = tuple(range(0.1, 10, 0.1))
        else:
            params['C'] = 1.0
            params['gamma'] = 1.0

    elif classifier_str == classifier_tuple[3]:
        classifier = CustomNaiveBayes()

    elif classifier_str == classifier_tuple[4]:
        params['voting'] = gui_params['voting']
        params['weights'] = gui_params['weights']
        classifier = Custom_Ensemble()
    
    # Tuning
    if gui_params['tuning'] and classifier_str != classifier_tuple[3] and classifier_str != classifier_tuple[4]:
        tuner = GridSearchCV(classifier, params, cv=10, n_jobs=-1)
        tuner.fit(train_x, train_y)
        params = tuner.best_params_
        print(params)

    classifier.set_params(**params) # Set params tuned previously or at runtime
    classifier.fit(train_x, train_y) # Train

    return classifier

def compute_performances(test_y, pred_y):
    cm = confusion_matrix(test_y, pred_y)
    eps = np.finfo(float).eps
    TP = cm[1,1]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    acc = (TP + TN) / (TP + TN + FP + FN + eps)
    TPR = TP / (TP + FN + eps)
    TNR = TN / (TN + FP + eps)
    FPR = FP / (TN + FP + eps)
    FNR = FN / (TP + FN + eps)
    p = TP / (TP + FP + eps)
    F1 = 2*TPR*p / (TPR+p + eps)
    return round(acc, 6), round(TPR, 6), round(TNR, 6), round(FPR, 6), round(FNR, 6), round(p, 6), round(F1, 6)