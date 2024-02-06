import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils import resample



class Custom_Ensemble:
    def __init__(self):
        pass

    def set_params (self, weights, voting, algorithm): 

        self.algorithm = algorithm
        if self.algorithm == 'forest':
            self.kNN_clf = DecisionTreeClassifier(max_depth=10, max_features=10, random_state=0)
            self.dTree_clf = DecisionTreeClassifier(max_depth=10, max_features=10, random_state=0)
            self.gNB_clf = DecisionTreeClassifier(max_depth=10, max_features=10, random_state=0)
        elif self.algorithm == 'boosting':
            self.kNN_clf = DecisionTreeClassifier(max_depth=10, max_features=10, random_state=0)
            self.dTree_clf = DecisionTreeClassifier(max_depth=10, max_features=10, random_state=0)
            self.gNB_clf = SVC(probability=True, kernel='rbf', C=1.5, gamma=0.5) 
        else:    
            self.kNN_clf = KNeighborsClassifier(n_neighbors=5)
            self.dTree_clf = DecisionTreeClassifier(max_depth=10, max_features=10, random_state=0)
            self.gNB_clf = SVC(probability=True, kernel='rbf', C=1.5, gamma=0.5)     
        self.voting = voting
        self.w = np.array(weights).astype(int)
        self.labels = [0, 1]
        self.wsum =  self.w.sum()
        

    def fit(self, x, y):
        if self.algorithm == 'bagging':
            x1 = []
            y1 = []
            for _ in range(3):
                xi, yi = resample(x, y)
                x1.append(xi)
                y1.append(yi)
            self.kNN_clf.fit(x1[0], y1[0])
            self.dTree_clf.fit(x1[1], y1[1])
            self.gNB_clf.fit(x1[2], y1[2])

        elif self.algorithm == 'standard':
            self.kNN_clf.fit(x, y)
            self.dTree_clf.fit(x, y)
            self.gNB_clf.fit(x, y)
            
        if self.algorithm == 'forest':
            x1 = []
            y1 = []
            self.feature_sets = []
            for _ in range(3):
                xi, yi = resample(x, y)
                x1.append(xi)
                y1.append(yi)

                # Seleziona casualmente il 70% degli attributi
                n_features = x.shape[1]
                subset_size = int(0.5 * n_features)  # Calcola il 70% del numero totale di attributi
                feature_indices = np.random.choice(n_features, size=subset_size, replace=False)
                self.feature_sets.append(feature_indices)
            self.kNN_clf.fit(x1[0][:, feature_indices], y1[0])
            self.dTree_clf.fit(x1[1][:, feature_indices], y1[1])
            self.gNB_clf.fit(x1[2][:, feature_indices], y1[2])


        elif self.algorithm == 'boosting':

            sample_weights = np.full(x.shape[0], (1. / x.shape[0]))

            for clf in [self.kNN_clf, self.dTree_clf, self.gNB_clf]:

                clf.fit(x, y, sample_weight=sample_weights)

                y_pred = clf.predict(x)
                incorrect = (y_pred != y)
                error = np.mean(np.average(incorrect, weights=sample_weights, axis=0))

                clf_weight = np.log((1. - error) / error) + np.log(2.)

                sample_weights *= np.exp(clf_weight * incorrect * ((sample_weights > 0) | (clf_weight > 0)))
                sample_weights /= np.sum(sample_weights)
                
        self.fitted = True


    def predict(self, test_x):
        proba = []
        if self.algorithm == 'forest':
            proba.append(self.kNN_clf.predict_proba(test_x[:, self.feature_sets[0]]))
            proba.append(self.dTree_clf.predict_proba(test_x[:, self.feature_sets[1]]))
            proba.append(self.gNB_clf.predict_proba(test_x[:, self.feature_sets[2]]))
        else:
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
        if self.algorithm == 'forest':
            proba.append(self.kNN_clf.predict_proba(test_x[:, self.feature_sets[0]]))
            proba.append(self.dTree_clf.predict_proba(test_x[:, self.feature_sets[1]]))
            proba.append(self.gNB_clf.predict_proba(test_x[:, self.feature_sets[2]]))
        else:
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
            
        return pred_y
    
