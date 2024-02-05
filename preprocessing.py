from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler,  normalize
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif, SequentialFeatureSelector
from imblearn.under_sampling import RandomUnderSampler, InstanceHardnessThreshold, NearMiss, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.neighbors import KNeighborsClassifier

imputation_tuple = ('Mean', 'Most frequent', 'Neighbors')
sampling_tuple = ('No sampling', 'Random without replacement', 'Random with replacement', 'Fixed stratified', 'Proportional stratified')
balancing_tuple = ('No balancing', 'Random undersampling', 'Probabilistic undersampling', 'Nearest to nearest', 'Nearest to farthest', 'Cluster Centroid', 'Random oversampling', 'Oversampling SMOTE', 'Oversampling ADASYN')
reduction_tuple = ('No dimensionality reduction', 'Principal Components Analysis', 'Sparse Random Projection', 'Gaussian Random Projection', 'Feature agglomeration', 'Variance threshold', 'Best chi2 score', 'Best mutual info score', 'Backword selection', 'Forward selection')
transformation_tuple = ('No transformation', 'Z-Score standardization', 'Min-Max standardization', 'L1 normalization', 'L2 normalization')

def pre_processing(X, y, imputation, transformation, reduction, balancing, sampling):
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
        reduct_obj = FeatureAgglomeration()
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
    if balancing == balancing_tuple[1]:
        balance_obj = RandomUnderSampler()
    elif balancing == balancing_tuple[2]:
        balance_obj = InstanceHardnessThreshold()
    elif balancing == balancing_tuple[3]:
        balance_obj = NearMiss(version=1)
    elif balancing == balancing_tuple[4]:
        balance_obj = NearMiss(version=2)
    elif balancing == balancing_tuple[5]:
        balance_obj = ClusterCentroids()
    elif balancing == balancing_tuple[6]:
        balance_obj = RandomOverSampler()
    elif balancing == balancing_tuple[7]:
        balance_obj = SMOTE()
    elif balancing == balancing_tuple[8]:
        balance_obj = ADASYN()
    else:
        balance_obj = None
    if balance_obj:
        X, y = balance_obj.fit_resample(X, y)

    # Sampling

    return X, y
