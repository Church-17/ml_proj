import sklearn as sl
import numpy as np
from sklearn.model_selection import train_test_split

class Custom_Ensemble:

    def __init__(self, parametri):
        self.estimators = ['Decision tree', 'K nearest neighbor', 'Support Vector Classifier']
        self.voting = parametri['voting']
        self.w = parametri['w']
        self.fitted = False
        
    def fit(self, x, y):
        self.labels = ['0', '1']
        classifier_params = {}
        for estimator in self.estimators:
            sub_train_x, _, sub_train_y, _ =  train_test_split(x, y, test_size=0.20, stratify=y)
            if estimator == 'Decision tree':
                classifier_params['option'] = 'Gini'
                classification(estimator, x, y, train_y, )
            elif  estimator == 'K nearest neighbor':
                classifier_params['option'] = 'Uniform'
                classification(estimator, x, y, train_y, classifier_params)
            elif  estimator == 'Support Vector Classifier':
                classifier_params['option'] = 'Linear'
                classification(estimator, x, y, train_y, classifier_params)

    
            estimator.fit(sub_train_x, sub_train_y)

        self.fitted = True

    def predict(self, test_x):
        if self.fitted:
            proba = []
            i = 0
            for estimator in self.estimators:
                proba.append(estimator.predict_proba(test_x) * self.w[i])
                i +=1
            proba = np.array(proba)
            pred_y = []
            for i in range(0, len(test_x)):
                if self.voting == 'hard':
                    # Hard voting
                    votes = [np.argmax(p) for p in proba[:,i,:].T]
                    pred_y.append(np.argmax(np.bincount(votes, weights=self.w)))
                elif self.voting == 'soft':
                    # Soft voting
                    pred_y.append(majority_voting(self.labels, proba[:,i,:].T, self.voting, self.w))
            return pred_y
        else:
            print("Il classificatore non è ancora stato addestrato")

def majority_voting(label, proba, voting, w=None):
    if voting == "hard":    #Hard voting
        votes = label[np.argmax(proba, axis=0)]# estrae i voti 
        if w is None: # peso uniforme
            unique, counts = np.unique(votes, return_counts=True) #conta i voti
            index = counts.argmax() # estrae il voto massimo
        else: # voti pesati
            w_votes = []
            for i, v in enumerate(votes):
                w_votes = w_votes + [v]* w[i]
            unique, counts = np.unique(w_votes, return_counts=True)
            index = counts.argmax()
        return unique[index]
    else: #soft voting
        if w is None: # peso uniforme
            s_prob = np.sum(proba, axis=0)
            index = s_prob.argmax()
        else: # voti pesati
            w_s_prob = np.sum(proba*w, axis=0)
            index = w_s_prob.argmax()
        return label[index]
    
