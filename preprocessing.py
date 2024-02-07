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
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Defining tuples for preprocessing tecniques
imputation_tuple = ('Mean', 'Most frequent', 'Neighbors')
sampling_tuple = ('No sampling', 'Random without replacement', 'Random with replacement', 'Fixed stratified', 'Proportional stratified')
undersampling_tuple = ('No undersampling', 'Random undersampling', 'Probabilistic undersampling', 'Nearest to nearest', 'Nearest to farthest', 'Cluster Centroid')
oversampling_tuple = ('No oversampling', 'Random oversampling', 'Oversampling SMOTE', 'Oversampling ADASYN')
reduction_tuple = ('No dimensionality reduction', 'Principal Components Analysis', 'Sparse Random Projection', 'Gaussian Random Projection', 'Feature agglomeration', 'Variance threshold', 'Best chi2 score', 'Best mutual info score', 'Backword selection', 'Forward selection', 'Correlation selection')
transformation_tuple = ('No transformation', 'Z-Score standardization', 'Min-Max standardization', 'L1 normalization', 'L2 normalization', 'Lmax normalization')

def pre_processing(X, y, imputation, transformation, reduction, undersampling, oversampling, sampling):
    "Performs preprocessing on the given data"

    # Selecting the tecnique to HANDLE MISSING VALUES
    if imputation == imputation_tuple[0]:
        impute_obj = SimpleImputer(strategy='mean')
    elif imputation == imputation_tuple[1]:
        impute_obj = SimpleImputer(strategy='most_frequent')
    elif imputation == imputation_tuple[2]:
        impute_obj = KNNImputer()
    else:
        impute_obj = None
    if impute_obj:
        X = impute_obj.fit_transform(X) # Trasforing the dataset according to the selected tecnique

    # Selecting the TRASFORMATION tecnique and applying the transformation to the dataset
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
    elif transformation == transformation_tuple[5]:
        X = normalize(X, 'max')

    # Selecting the DIMENSIONALITY REDUCTION tecnique
    new_n_features = 48
    if reduction == reduction_tuple[1]:
        reduct_obj = PCA(new_n_features)
    elif reduction == reduction_tuple[2]:
        reduct_obj = SparseRandomProjection(new_n_features)
    elif reduction == reduction_tuple[3]:
        reduct_obj = GaussianRandomProjection(new_n_features)
    elif reduction == reduction_tuple[4]:
        reduct_obj = FeatureAgglomeration(new_n_features)
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
    elif reduction == reduction_tuple[10]:  # Correlation-based feature selection
        correlation_matrix = DataFrame(X).corr()
        corr_np = correlation_matrix.to_numpy()
        corr_attr = np.concatenate((np.where(corr_np>0.90), np.where(corr_np<-0.90)), axis=1)   # Correlation threshold = 0.9
        del_attr = [attr1 for attr0, attr1 in zip(corr_attr[0], corr_attr[1]) if attr1 > attr0]
        unique_del = np.unique(del_attr)
        X = np.delete(X, unique_del, 1) # Filtering dataset: removing features with high correlation
        reduct_obj = None
    else:
        reduct_obj = None
    if reduct_obj:
        X = reduct_obj.fit_transform(X, y)  # Trasforing the dataset according to the selected tecnique
    
    # BALANCING
    if oversampling == oversampling_tuple[0]: # If only undersampling is requested
        under_ratio = 'auto'
    else:                                     # If both undersampling and oversampling are requested
        under_ratio = 2 * len(y[y==1]) / len(y)     # Proportion of dataset to undersample
    
    # Selecting the UNDERSAMPLING tecnique
    if undersampling == undersampling_tuple[1]:
        balance_obj = RandomUnderSampler(sampling_strategy=under_ratio)
    elif undersampling == undersampling_tuple[2]:
        balance_obj = InstanceHardnessThreshold(sampling_strategy=under_ratio)
    elif undersampling == undersampling_tuple[3]:
        balance_obj = NearMiss(sampling_strategy=under_ratio, version=1)
    elif undersampling == undersampling_tuple[4]:
        balance_obj = NearMiss(sampling_strategy=under_ratio, version=2)
    elif undersampling == undersampling_tuple[5]:
        balance_obj = ClusterCentroids(sampling_strategy=under_ratio, estimator=KMeans(n_init=10))
    else:
        balance_obj = None
    if balance_obj:
        X, y = balance_obj.fit_resample(X, y)   # Trasforing the dataset according to the selected tecnique

    if undersampling == undersampling_tuple[0]: # If only oversampling is requested
        over_ratio = 'auto'
    else:                                       # If both undersampling and oversampling are requested
        over_ratio = 1                          # Now the proportion to oversample is 1 (half oversampled, half undersampled)

    # Selecting the OVERSAMPLING tecnique
    if oversampling == oversampling_tuple[1]:
        balance_obj = RandomOverSampler(sampling_strategy=over_ratio)
    elif oversampling == oversampling_tuple[2]:
        balance_obj = SMOTE(sampling_strategy=over_ratio)
    elif oversampling == oversampling_tuple[3]:
        balance_obj = ADASYN(sampling_strategy=over_ratio)
    else:
        balance_obj = None
    if balance_obj:
        X, y = balance_obj.fit_resample(X, y)   # Trasforing the dataset according to the selected tecnique

    # Selecting the SAMPLING tecnique and sampling the dataset
    sample_dim = (len(y[y == 0]) + len(y[y == 1])) // 2
    if sampling == sampling_tuple[1]:
        X, y = resample(X, y, n_samples=sample_dim, replace=False, stratify=None)

    elif sampling == sampling_tuple[2]:
        X, y = resample(X, y, n_samples=sample_dim, replace=True, stratify=None)
        
    elif sampling == sampling_tuple[3]:
        X, y = resample(X, y, n_samples=sample_dim, replace=False, stratify=y)
    
    elif sampling == sampling_tuple[4]:
        x1, y1 = resample(X[y == 0], y[y == 0], n_samples=int(np.ceil((len(y[y == 0]) / len(y)) * sample_dim)))
        x2, y2 = resample(X[y == 1], y[y == 1], n_samples=int(np.ceil((len(y[y == 1]) / len(y)) * sample_dim)))

        X = np.concatenate((x1, x2))
        y = np.concatenate((y1, y2))

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0, test_size=0.25)   # Train - test split after preprocessing
    return train_x, test_x, train_y, test_y
