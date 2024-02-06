import numpy as np
from classification2 import classify

class Custom_Ensemble:
    def __init__(self, weights, voting):
        self.estimators = {
            'Decision tree': 0,
            'K nearest neighbor': 0,
            'Support Vector Classifier': 0
        }        
        self.voting = voting
        self.w = weights
        self.fitted = False
        self.wsum = weights.sum()
        self.labels = ['0', '1']
        

def fit(self, x, y):
    classifier_params = {}
    for key in self.estimators.keys():
        if key == 'Decision tree':
            classifier_params['option'] = 'Gini'
        elif key == 'K nearest neighbor':
            classifier_params['option'] = 'Uniform'
        elif key == 'Support Vector Classifier':
            classifier_params['option'] = 'Polinomial'
        self.estimators[key] = classify(key, x, y, classifier_params)
    self.fitted = True


def predict(self, test_x):
    proba = []
    for key in self.estimators.keys():
        if key == 'Support Vector Classifier':
            proba.append(self.estimators[key].decision_function(test_x))
        else:    
            proba.append(self.estimators[key].predict_proba(test_x))

    proba = np.array(proba)
    pred_y = []

    for i in range(0, len(test_x)):
        if self.voting == 'hard':
            votes = [np.argmax(p) for p in proba[:,i,:].T]
            w_votes = []
            for i, v in enumerate(votes):
                w_votes = w_votes + [v] * self.w[i]
            unique, counts = np.unique(w_votes, return_counts=True)
            index = counts.argmax()
            pred_y.append(unique[index])
        elif self.voting == 'soft':
            w_s_prob = np.sum(proba[:,i,:]*self.w, axis=0)
            index = w_s_prob.argmax()
            pred_y.append(self.labels[index])
    return pred_y