import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import resample


class Custom_Ensemble:

    # Initializing the classificator
    def __init__(self):
        pass


    def set_params (self, weights, voting, algorithm): 
        "Initializes the classificators according to the specific approach and the selected parameters"

        self.algorithm = algorithm

        # Performs random forest ensemble
        if self.algorithm == 'forest':
            self.classificatore_1 = DecisionTreeClassifier(max_depth=None, criterion='entropy', min_samples_leaf=3, min_samples_split=2)
            self.classificatore_2 = DecisionTreeClassifier(max_depth=None, criterion='entropy', min_samples_leaf=3, min_samples_split=2)
            self.classificatore_3 = DecisionTreeClassifier(max_depth=None, criterion='entropy', min_samples_leaf=3, min_samples_split=2)

        # Performs standard ensemble (with bagging, boosting, majority voting)
        else:
            self.classificatore_1 = DecisionTreeClassifier(max_depth=None, criterion='entropy', min_samples_leaf=3, min_samples_split=2)
            self.classificatore_2 = GaussianNB()
            self.classificatore_3 = SVC(probability=True, kernel='rbf', C=1.7, gamma=1.4)  

        # Sets the params for the ensemble
        self.voting = voting                    # voting policy (hard or soft)
        self.w = np.array(weights).astype(int)  # weights
        self.labels = [0, 1]                    # class labels
        self.wsum =  self.w.sum()               # sum of the weights
        
    def fit(self, X, y):
        "Training of the classifier"

        # If MAJORITY VOTING is selected
        if self.algorithm == 'standard':
            self.classificatore_3.fit(X, y)
            self.classificatore_1.fit(X, y) # Training of each classifier
            self.classificatore_2.fit(X, y)
            
        # If BAGGING is selected
        elif self.algorithm == 'bagging':
            x1 = [] # Bootstrap samples
            y1 = [] # Labels of the samples

            for _ in range(3): # Creating the sample training sets
                xi, yi = resample(X, y)
                x1.append(xi) 
                y1.append(yi)

            self.classificatore_1.fit(x1[0], y1[0]) # addestro i 3 classificatori con i 3 campioni di bootstrap distinti
            self.classificatore_2.fit(x1[1], y1[1])
            self.classificatore_3.fit(x1[2], y1[2])



        # If BOOSTING is selected
        elif self.algorithm == 'boosting':

            sample_weights = np.full(X.shape[0], (1. / X.shape[0])) # Initializing uniform weights for each sample record

            for clf in [self.classificatore_1, self.classificatore_2, self.classificatore_3]: # For each classifier...

                clf.fit(X, y, sample_weight=sample_weights) # Train it with the current training weights

                # Performing predictions with the classifier and adjusting record's weights and classifier's weight according to its performances 
                y_pred = clf.predict(X) 
                incorrect = (y_pred != y)
                error = np.mean(np.average(incorrect, weights=sample_weights, axis=0))
                clf_weight = np.log((1. - error) / error) + np.log(2.)
                sample_weights *= np.exp(clf_weight * incorrect * ((sample_weights > 0) | (clf_weight > 0)))
                sample_weights /= np.sum(sample_weights)
                

        # If RANDOM FOREST is selected
        if self.algorithm == 'forest':
            x1 = [] # Random forest samples
            y1 = []
            self.feature_sets = [] # List for the random subset of featutures

            for _ in range(3): # Creating the sample training set for each random tree
                xi, yi = resample(X, y) 
                x1.append(xi)
                y1.append(yi)

                # Extracting the random subset of features
                n_features = X.shape[1] 
                subset_size = int(0.85 * n_features) 
                feature_indices = np.random.choice(n_features, size=subset_size, replace=False)
                self.feature_sets.append(feature_indices)

            # Training the trees on the basis of the random training sample and the random feature subset
            self.classificatore_1.fit(x1[0][:, feature_indices], y1[0])
            self.classificatore_2.fit(x1[1][:, feature_indices], y1[1])
            self.classificatore_3.fit(x1[2][:, feature_indices], y1[2])

    def predict(self, test_x):
        "Predicts the target labels"

        proba = []  # Matrix of probabilities

        if self.algorithm == 'forest':  # If random forest is selected
            
            # Predicting the probabilities on test set with each tree
            proba.append(self.classificatore_1.predict_proba(test_x[:, self.feature_sets[0]]))
            proba.append(self.classificatore_2.predict_proba(test_x[:, self.feature_sets[1]]))
            proba.append(self.classificatore_3.predict_proba(test_x[:, self.feature_sets[2]]))
        
        else: # Otherwise, uses the standard ensemble
            proba.append(self.classificatore_1.predict_proba(test_x))
            proba.append(self.classificatore_2.predict_proba(test_x))
            proba.append(self.classificatore_3.predict_proba(test_x))

        pred_y = np.zeros([len(test_x),1]) # List of the predicted classes

        voting = np.zeros([len(test_x),2]) # Matrix of the probabilities for both classes 

        for i in range(0, len(test_x)): # For each test record...
            for j in range(0,3):  # For each classifier...
                if self.voting == 'hard':  # If hard voting is selected

                    # Adding a vote for the class with the higher probability considering the weight of each classifier
                    if proba[j][i][1] > proba[j][i][0]:
                        voting[i][1] += (1 * self.w[j] / self.wsum)
                    else:
                        voting[i][0] += (1 * self.w[j]/ self.wsum)
                
                else: # If soft voting is selected

                    # Adding the probability estimated for each of the classes considering the weight of each classifier
                    voting[i][0] += (float(proba[j][i][0]) * float(self.w[j]/ self.wsum))
                    voting[i][1] += (float(proba[j][i][1]) * float(self.w[j]/ self.wsum))

                    
            pred_y[i] = (self.labels[np.argmax(voting[i][:])]) # The predicted class is the one if the higher probability

        return pred_y 
    
    def predict_proba(self, test_x):
        "Predicting the probabilities for each class"

        proba = [] # Initializes the matrix of probabilities

        if self.algorithm == 'forest': # If random forest is selected
            proba.append(self.classificatore_1.predict_proba(test_x[:, self.feature_sets[0]])) #faccio la previsione considerando solo il sottoinsieme di attirbuti con cui è stato addestrato l'albero
            proba.append(self.classificatore_2.predict_proba(test_x[:, self.feature_sets[1]]))
            proba.append(self.classificatore_3.predict_proba(test_x[:, self.feature_sets[2]]))
        
        else:  # Otherwise, uses the standard ensemble to compute the probabilities
            proba.append(self.classificatore_1.predict_proba(test_x))
            proba.append(self.classificatore_2.predict_proba(test_x))
            proba.append(self.classificatore_3.predict_proba(test_x))


        pred_y = np.zeros([len(test_x),2]) # Initializing the matrix of the predicted classes

        voting = np.zeros([len(test_x), 2]) # Initializing the matrix with the votes of each classifier 

        for i in range(0, len(test_x)): # For each test record...
            for j in range(0,3): # For each classifier...

                # Adding the probability estimated for each of the classes considering the weight of each classifier
                voting[i][0] += (float(proba[j][i][0]) * float(self.w[j]/ self.wsum)) 
                voting[i][1] += (float(proba[j][i][1]) * float(self.w[j]/ self.wsum))  

            pred_y[i][0] = voting[i][0]
            pred_y[i][1] = voting[i][1]
            
        return pred_y


