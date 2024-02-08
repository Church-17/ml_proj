import numpy as np

class CustomNaiveBayes(object):
    "Custom Naive Bayes Classifier"

    def __init__(self):
        self.probabilities = None
        self.pred_y = None

    def fit(self, train_x, train_y):
        "Training of the classifier"

        _, counts = np.unique(train_y, return_counts=True)
        self.prob_neg = counts[0] / len(train_y)    # Computing a priori probabilities
        self.prob_pos = counts[1] / len(train_y)
        self.n_attr = train_x.shape[1]
        self.neg_means = np.zeros(self.n_attr)      # Initializing arrays for means and variances
        self.neg_variances = np.zeros(self.n_attr)
        self.pos_means = np.zeros(self.n_attr)
        self.pos_variances = np.zeros(self.n_attr)

        for i in range(self.n_attr):                # Computing mean and variance of positive and negative classes
            self.neg_means[i] = np.mean(train_x[:,i][train_y == 0])
            self.neg_variances[i] = np.var(train_x[:,i][train_y == 0])
            self.pos_means[i] = np.mean(train_x[:,i][train_y == 1])
            self.pos_variances[i] = np.var(train_x[:,i][train_y == 1])


    def predict(self, test_x):
        "Predicts the target labels"

        self.probabilities = np.zeros((len(test_x), 2))
        self.pred_y = np.zeros(len(test_x))

        # Computing the conditioned probabilities
        for obj in range(len(test_x)):
            pos_probability = self.prob_pos
            neg_probability = self.prob_neg
            record = test_x[obj]
            # Computing numerators of probabilities to belong to positive or negative classes
            for attr in range(self.n_attr):
                neg_probability *= (1/np.sqrt(2*np.pi*self.neg_variances[attr])) * np.exp(-1 * ((record[attr] - self.neg_means[attr])**2 / (2 * self.neg_variances[attr])))
                pos_probability *= (1/np.sqrt(2*np.pi*self.pos_variances[attr])) * np.exp(-1 * ((record[attr] - self.pos_means[attr])**2 / (2 * self.pos_variances[attr])))
            self.probabilities[obj][0] = neg_probability
            self.probabilities[obj][1] = pos_probability
            # Assigning the class by comparing numerators of probabilities
            if self.probabilities[obj][1] > self.probabilities[obj][0]:
                self.pred_y[obj] = 1

        return self.pred_y

    def predict_proba(self, test_x):
        "Probabilities to belong to the classes"

        assert self.probabilities is not None
        for obj in range(len(test_x)):              # Computing real probabilities of belonging to positive or negative class
            sum = self.probabilities[obj].sum()
            if sum == 0:                            # If the sum of probabilities is computed as 0, it is assigned as an infinitesimal
                sum = np.finfo(float).eps
            self.probabilities[obj][0] /= sum
            self.probabilities[obj][1] /= sum
        
        return self.probabilities                   # Returning real probabilities

    def score(self, test_x, test_y):
        "Computes the accuracy of the classifier"

        assert self.pred_y is not None
        return (self.pred_y == test_y).sum() / len(test_y)
    
    def set_params(self, **params):
        "Setting of the hyperparameters"
        return 
    
