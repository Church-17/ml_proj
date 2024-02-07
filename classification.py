from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from custom_naive_bayes import CustomNaiveBayes
from custom_ensemble import Custom_Ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np

# Defining tuples for classifiers' options
classifier_tuple = ('Decision tree', 'K nearest neighbor', 'Support Vector Classifier', 'Custom Naive Bayes', 'Custom ensamble')
weights_tuple = ('Uniform', 'Distance')
distance_tuple = ('Euclidean', 'Manhattan', 'Cosine', 'Pearson correlation')
purity_tuple = ('Gini', 'Entropy', 'LogLoss')
kernel_tuple = ('Linear', 'Polinomial', 'RBF')

def init_classification(classifier_str, gui_params):
    "Initializes the classifier with tuned hyperparameters"
    params = {}

    # Decision Tree
    if classifier_str == classifier_tuple[0]:
        classifier = DecisionTreeClassifier()
        if gui_params['tuning']:        # Real time tuning
            params['criterion'] = ('gini', 'entropy', 'log_loss')
            params['max_depth'] = [None] + list(range(2, 8))
            params['min_samples_leaf'] = tuple(range(1, 5))
            params['min_samples_split'] = tuple(range(2, 5))
        else:                       # Selecting purity metric
            if gui_params['option1'] == purity_tuple[0]:
                params['criterion'] = 'gini'
            elif gui_params['option1'] == purity_tuple[1]:
                params['criterion'] = 'entropy'
            elif gui_params['option1'] == purity_tuple[2]:
                params['criterion'] = 'log_loss'
            params['max_depth'] = 6     # Imputing tuned hyperparameters
            params['min_samples_leaf'] = 3
            params['min_samples_split'] = 2

    # K-Nearest Neighbour
    elif classifier_str == classifier_tuple[1]:
        classifier = KNeighborsClassifier()
        if gui_params['tuning']:    # Real time tuning
            params['n_neighbors'] = tuple(range(1, 20))
            params['weights'] = ('uniform','distance')
            params['metric'] = ('euclidean', 'manhattan', 'cosine', 'correlation')
        else:                       # Selecting distance metric
            if gui_params['option1'] == distance_tuple[0]:
                params['metric'] = 'euclidean'
            elif gui_params['option1'] == distance_tuple[1]:
                params['metric'] = 'manhattan'
            elif gui_params['option1'] == distance_tuple[2]:
                params['metric'] = 'cosine'
            elif gui_params['option1'] == distance_tuple[3]:
                params['metric'] = 'correlation'
            if gui_params['option2'] == 0:
                params['weights'] = 'uniform'
            elif gui_params['option2'] == 1:
                params['weights'] = 'distance'
            params['n_neighbors'] = 2  # Imputing tuned number of neighbours

    # Support Vector Machine
    elif classifier_str == classifier_tuple[2]:
        classifier = SVC(probability=True)
        if gui_params['tuning']:    # Real time tuning
            params['kernel'] = ['linear', 'poly', 'rbf']
            params['C'] = tuple([float(x)/10 for x in range(10, 30)])
            params['gamma'] = tuple([float(x)/10 for x in range(0, 10)])
        else:                       # Selecting kernel
            if gui_params['option1'] == kernel_tuple[0]:
                params['kernel'] = 'linear'
            elif gui_params['option1'] == kernel_tuple[1]:
                params['kernel'] = 'poly'
            elif gui_params['option1'] == kernel_tuple[2]:
                params['kernel'] = 'rbf'
            params['C'] = 1.5       # Imputing tuned hyperparameters
            params['gamma'] = 0.5

    # Custom Naive Bayes
    elif classifier_str == classifier_tuple[3]:
        classifier = CustomNaiveBayes()

    # Custom Ensemble
    elif classifier_str == classifier_tuple[4]:
        params['voting'] = gui_params['voting']         # Imputing Voting policy
        params['weights'] = gui_params['weights']       # Imputing Weights
        params['algorithm'] = gui_params['algorithm']   # Imputing ensemble training algorithm
        classifier = Custom_Ensemble()

    return classifier, params

def tuning(classifier, params, train_x, train_y):
    "Tuning of the hyperparameters of a classifier"

    tuner = GridSearchCV(classifier, params, cv=5, n_jobs=-1)   # Defining object to tune the hyperparameters
    tuner.fit(train_x, train_y)                                 # Fitting the tuner
    print(tuner.best_params_)                                   # Output: Optimal hyperparameters
    return tuner.best_params_

def compute_performances(test_y, pred_y):
    "Computes the performances of a classifier"

    cm = confusion_matrix(test_y, pred_y)                       # Computing the confusion matrix
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
    return round(acc, 6), round(TPR, 6), round(TNR, 6), round(FPR, 6), round(FNR, 6), round(p, 6), round(F1, 6) # Output: Evaluation metrics of the classifier