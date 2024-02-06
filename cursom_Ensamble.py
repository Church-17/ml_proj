import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import resample



class Custom_Ensemble:
    def __init__(self):
        self.kNN_clf = KNeighborsClassifier(n_neighbors=5)
        self.dTree_clf = DecisionTreeClassifier(max_depth=10, max_features=10, random_state=0)
        self.gNB_clf = SVC(probability=True, kernel='rbf', C=1.5, gamma=0.5)


    def set_params (self, weights, voting, bagging):      
        self.voting = voting
        self.w = np.array(weights).astype(int)
        self.labels = [0, 1]
        self.wsum =  self.w.sum()
        self.bagging = bagging
        

    def fit(self, x, y):
        if self.bagging == True:
            x=[]
            y=[]
            for i in range(3):
                x[i], y[i] = resample(x, y)
            self.kNN_clf.fit(x[i],y[i])
            self.dTree_clf.fit(x[i],y[i])
            self.gNB_clf.fit(x[i],y[i])
        else:
            self.kNN_clf.fit(x,y)
            self.dTree_clf.fit(x,y)
            self.gNB_clf.fit(x,y)
        self.fitted = True


    def predict(self, test_x):
        proba = []
        proba.append(self.kNN_clf.predict_proba(test_x))
        proba.append(self.dTree_clf.predict_proba(test_x))
        proba.append(self.gNB_clf.predict_proba(test_x))

        pred_y = np.zeros([len(test_x),1])
        voting = np.zeros([len(test_x),2])



        for i in range(0, len(test_x)):
            for j in range(0,3):
                if self.voting == 'hard':
                    if proba[j][i][1] > proba[j][i][0]:
                        voting[i][1] += (1 * self.w[j] / self.wsum)
                    else:
                        voting[i][0] += (1 * self.w[j]/ self.wsum)
                else: 
                    voting[i][0] += (float(proba[j][i][0]) * float(self.w[j]/ self.wsum))
                    voting[i][1] += (float(proba[j][i][1]) * float(self.w[j]/ self.wsum))

                    
            pred_y[i] = (self.labels[np.argmax(voting[i][:])])

        print(self.voting, ": ")
        for i in range(0,10):
            print(voting[i])
        return pred_y 
    

    def predict_proba(self, test_x):
        proba = []
        proba.append(self.kNN_clf.predict_proba(test_x))
        proba.append(self.dTree_clf.predict_proba(test_x))
        proba.append(self.gNB_clf.predict_proba(test_x))

        pred_y = np.zeros([len(test_x),2])

        voting = np.zeros([len(test_x), 2])

        for i in range(0, len(test_x)):
            for j in range(0,3):
                voting[i][0] += (float(proba[j][i][0]) * float(self.w[j]/ self.wsum))
                voting[i][1] += (float(proba[j][i][1]) * float(self.w[j]/ self.wsum))  
            pred_y[i][0] = voting[i][0]
            pred_y[i][1] = voting[i][1]
           #pred_y.append([voting[i][0],voting[i][1]])
            
        return pred_y