from scipy.io import arff
from pandas import DataFrame

# Function to load dataset
def load_dataset(path):
    dataset_raw = arff.loadarff(path) # Load ARFF file
    df = DataFrame(dataset_raw[0]) # Convert to Pandas DataFrame
    return df

def split_attrib_class(df: DataFrame):
    X = df.iloc[:,:-1] # Retrieve attribute matrix
    y = df.iloc[:,-1].to_numpy(dtype=int)[:,] # Retrieve class array
    return X, y