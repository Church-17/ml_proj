from scipy.io import arff
import pandas as pd

def load_dataset(path):
    dataset_raw = arff.loadarff(path)
    df = pd.DataFrame(dataset_raw[0])
    return df

def split_attrib_class(df: pd.DataFrame):
    X=df.iloc[:,:-1] 
    Y=df.iloc[:,-1].to_numpy(dtype=int)[:,]
    return X, Y