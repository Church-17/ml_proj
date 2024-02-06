from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler,  normalize
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif, SequentialFeatureSelector
from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, NearMiss, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import numpy as np
from numpy.random import choice
from sklearn.model_selection import train_test_split

imputation_tuple = ('Mean', 'Most frequent', 'Neighbors')
sampling_tuple = ('No sampling', 'Random without replacement', 'Random with replacement', 'Fixed stratified', 'Proportional stratified')
undersampling_tuple = ('No undersampling', 'Random undersampling', 'Probabilistic undersampling', 'Nearest to nearest', 'Nearest to farthest', 'Cluster Centroid')
oversampling_tuple = ('No oversampling', 'Random oversampling', 'Oversampling SMOTE', 'Oversampling ADASYN')
reduction_tuple = ('No dimensionality reduction', 'Principal Components Analysis', 'Sparse Random Projection', 'Gaussian Random Projection', 'Feature agglomeration', 'Variance threshold', 'Best chi2 score', 'Best mutual info score', 'Backword selection', 'Forward selection')
transformation_tuple = ('No transformation', 'Z-Score standardization', 'Min-Max standardization', 'L1 normalization', 'L2 normalization')

def pre_processing(X, y, imputation, transformation, reduction, undersampling, oversampling, sampling):
    # Handle missing values
    if imputation == imputation_tuple[0]:
        impute_obj = SimpleImputer(strategy='mean')
    elif imputation == imputation_tuple[1]:
        impute_obj = SimpleImputer(strategy='most_frequent')
    elif imputation == imputation_tuple[2]:
        impute_obj = KNNImputer()
    else:
        impute_obj = None
    if impute_obj:
        X = impute_obj.fit_transform(X)

    # Transformation
    if transformation == transformation_tuple[1]:
        zScore = StandardScaler()
        X = zScore.fit_transform(X)
    elif transformation == transformation_tuple[2]:
        MinMax = MinMaxScaler()
        X = MinMax.fit_transform(X)
    elif transformation == transformation_tuple[3]:
        X = normalize(X, 'l1')
    elif transformation == transformation_tuple[4]:
        X = normalize(X, 'l2')

    # Dimensionality
    new_n_features = 48
    if reduction == reduction_tuple[1]:
        reduct_obj = PCA()
    elif reduction == reduction_tuple[2]:
        reduct_obj = SparseRandomProjection(new_n_features)
    elif reduction == reduction_tuple[3]:
        reduct_obj = GaussianRandomProjection(new_n_features)
    elif reduction == reduction_tuple[4]:
        reduct_obj = FeatureAgglomeration(n_clusters=5)
    elif reduction == reduction_tuple[5]:
        reduct_obj = VarianceThreshold(1)
    elif reduction == reduction_tuple[6]:
        reduct_obj = SelectKBest(chi2, k=new_n_features)
    elif reduction == reduction_tuple[7]:
        reduct_obj = SelectKBest(mutual_info_classif, k=new_n_features)
    elif reduction == reduction_tuple[8]:
        reduct_obj = SequentialFeatureSelector(KNeighborsClassifier(), n_features_to_select=new_n_features, n_jobs= -1, direction='backward')
    elif reduction == reduction_tuple[9]:
        reduct_obj = SequentialFeatureSelector(KNeighborsClassifier(), n_features_to_select=new_n_features, n_jobs= -1, direction='forward')
    else:
        reduct_obj = None
    if reduct_obj:
        X = reduct_obj.fit_transform(X, y)
    
    # Balancing
    if oversampling == oversampling_tuple[0]:
        under_ratio = 'auto'
    else:
        under_ratio = 2 * len(y[y==1]) / len(y)
    
    if undersampling == undersampling_tuple[1]:
        balance_obj = RandomUnderSampler(sampling_strategy=under_ratio)
    elif undersampling == undersampling_tuple[2]:
        balance_obj = InstanceHardnessThreshold(sampling_strategy=under_ratio)
    elif undersampling == undersampling_tuple[3]:
        balance_obj = NearMiss(sampling_strategy=under_ratio, version=1)
    elif undersampling == undersampling_tuple[4]:
        balance_obj = NearMiss(sampling_strategy=under_ratio, version=2)
    elif undersampling == undersampling_tuple[5]:
        balance_obj = ClusterCentroids(sampling_strategy=under_ratio, estimator=KMeans(n_init='auto'))
    else:
        balance_obj = None
    if balance_obj:
        X, y = balance_obj.fit_resample(X, y)

    if undersampling == undersampling_tuple[0]:
        over_ratio = 'auto'
    else:
        over_ratio = 1

    if oversampling == oversampling_tuple[1]:
        balance_obj = RandomOverSampler(sampling_strategy=over_ratio)
    elif oversampling == oversampling_tuple[2]:
        balance_obj = SMOTE(sampling_strategy=over_ratio)
    elif oversampling == oversampling_tuple[3]:
        balance_obj = ADASYN(sampling_strategy=over_ratio)
    else:
        balance_obj = None
    if balance_obj:
        X, y = balance_obj.fit_resample(X, y)

    sample_dim = (len(y[y == 0]) + len(y[y == 1])) // 2
    sampled_X = [0] * sample_dim
    sampled_y = [0] * sample_dim
    if sampling == sampling_tuple[1]:
        sampling_obj = choice(len(y), sample_dim)
        for i in range(sample_dim):
            sampled_X[i] = X[sampling_obj[i]][:]
            sampled_y[i] = y[sampling_obj[i]][:]

    elif sampling == sampling_tuple[2]:
        sampling_obj = choice(len(y), sample_dim, replace=True)
        for i in range(sample_dim):
            sampled_X[i] = X[sampling_obj[i]][:]
            sampled_y[i] = y[sampling_obj[i]][:]
        X = sampled_X
        y = sampled_y

    elif sampling == sampling_tuple[3]:
        train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25, stratify=y)
        return train_x, test_x, train_y, test_y
    
    elif sampling == sampling_tuple[4]:
        sampling_obj_neg = choice(len(y[y == 0]), np.ceil((len(y[y == 0]) / len(y)) * sample_dim))
        sampling_obj_pos = choice(len(y[y == 1]), len(y) - np.ceil((len(y[y == 0]) / len(y)) * sample_dim))
    
        X = X[y in sampling_obj_neg or y in sampling_obj_pos]
        y = y[y in sampling_obj_neg or y in sampling_obj_pos]
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25, stratify=y)
    return train_x, test_x, train_y, test_y
