import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB




class Custom_Ensemble:
    def __init__(self):
        self.kNN_clf = KNeighborsClassifier(1)
        self.dTree_clf = DecisionTreeClassifier(random_state=0)
        self.gNB_clf = GaussianNB()

    def set_params (self, weights, voting):      
        self.voting = voting
        self.w = np.array(weights)
        self.labels = ['0', '1']
        

    def fit(self, x, y):
        self.kNN_clf.fit(x,y)
        self.dTree_clf.fit(x,y)
        self.gNB_clf.fit(x,y)
        self.fitted = True

def predict(self, test_x):
    
    proba1 = (self.kNN_clf.predict_proba(test_x))
    proba2 = (self.dTree_clf.predict_proba(test_x))
    proba3 = (self.gNB_clf.predict_proba(test_x))

    proba1_pos = proba1[:][1]
    proba1_neg = proba1[:][0]

    proba2_pos = proba2[:][1]
    proba2_neg = proba2[:][0]

    proba3_pos = proba3[:][1]
    proba3_neg = proba3[:][0]

    pred_y = []
    voting = np.zeros([len(test_x), 2])

    for i in range(0, len(test_x)):

        if self.voting == 'hard':
            if proba1_pos[i] > proba1_neg[i]:
                voting[i][1] += 1 *self.w[0]
            else:
                voting[i][0] += 1 *self.w[0]

            if proba2_pos[i] > proba2_neg[i]:
                voting[i][1] += 1 *self.w[1]
            else:
                voting[i][0] += 1 *self.w[1]

            if proba3_pos[i] > proba3_neg[i]:
                voting[i][1] += 1 *self.w[2]
            else:
                voting[i][0] += 1 *self.w[2]
                
        elif self.voting == 'soft': 
                
                voting[i][1] += proba1_pos *self.w[0]
                voting[i][0] += proba1_neg

                voting[i][1] += proba2_pos *self.w[1]
                voting[i][0] += proba2_neg *self.w[1]

                voting[i][1] += proba3_pos *self.w[2]
                voting[i][0] += proba3_neg *self.w[2]

        pred_y[i] = self.labels[np.argmax(voting[i][:], axis=1)]
               
    return pred_y 