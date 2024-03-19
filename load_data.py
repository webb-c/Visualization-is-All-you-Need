import numpy as np

def load_ndarray(file_path:str):
    data = np.load(file_path)
    
    return data
