from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

classifier_tuple = ('Ensamble classifier', 'Decision tree', 'K nearest neighbor')
distance_tuple = ('Uniform', 'Euclidean', 'Manhattan', 'Cosine', 'Pearson correlation')
purity_tuple = ('Gini', 'Entropy', 'LogLoss')

def init_classification(classifier_str, dataset, gui_params):
    params = {}

    if classifier_str == classifier_tuple[1]:
        if gui_params['option'] == purity_tuple[0]:
            params['criterion'] = 'gini'
        elif gui_params['option'] == purity_tuple[1]:
            params['criterion'] = 'entropy'
        elif gui_params['option'] == purity_tuple[2]:
            params['criterion'] = 'log_loss'
        params['max_depth'] = None
        params['max_features'] = "sqrt"
        params['min_samples_leaf'] = 10
        params['min_samples_split'] = 10
        classifier = DecisionTreeClassifier(**params)

    elif classifier_str == classifier_tuple[2]:
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
        params['n_neighbors'] = 10
        classifier = KNeighborsClassifier(**params)

    X = dataset.data
    Y = dataset.target
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.25)

    classifier.fit(train_x, train_y)

    pred_y = classifier.predict(test_x)