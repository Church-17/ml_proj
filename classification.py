from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

classifier_tuple = ('Ensamble classifier', 'Decision tree', 'K nearest neighbor')
purity_tuple = ('Euclidean', 'Manhattan', 'Chebychev', 'Tuning')
distance_tuple = ('Gini', 'Entropy', 'Classification error', 'Tuning')

def start_classification(classifier_str, dataset):
    if classifier_str == classifier_tuple[1]:
        classifier_obj = DecisionTreeClassifier
    elif classifier_str == classifier_tuple[2]:
        classifier_obj = KNeighborsClassifier

    X = dataset.data
    Y = dataset.target
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)
    classifier = classifier_obj()
    return classifier, train_x, test_x, train_y, test_y

def tuning(classifier: DecisionTreeClassifier | KNeighborsClassifier, params, train_x, train_y):
    tuner = GridSearchCV(classifier, params, cv=10, n_jobs=-1)
    tuner.fit(train_x, train_y)
    classifier.set_params(**tuner.best_params_)

def training(classifier: DecisionTreeClassifier | KNeighborsClassifier, train_x, train_y):
    classifier.fit(train_x, train_y)

def predicting(classifier: DecisionTreeClassifier | KNeighborsClassifier, test_x):
    return classifier.predict(test_x)