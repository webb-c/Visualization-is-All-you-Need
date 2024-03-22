import os
import numpy as np
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator

def load_ndarray(file_path:str):
    data = np.load(file_path)
    
    return data


def load_csv(file_path:str):
    data = pd.read_csv(file_path)
    
    return data



def load_log(log_dir, sort_by=None):
    """주어진 path에 있는 log를 읽어서, 각각의 tag를 column으로 하고 row에는 step별 value가 기록된 DataFrame을 만들어 반환합니다.
    #! This code reference from https://gist.github.com/laszukdawid/62656cf7b34cac35b325ba21d46ecfcd 
    Args:
        log_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.
    
    Returns:
        pandas.DataFrame with columns as tags and rows as values.
    """
    def convert_tfevent(filepath):
        data = {}
        for e in summary_iterator(filepath):
            if len(e.summary.value) > 0:
                for value in e.summary.value:
                    tag = value.tag
                    if tag not in data:
                        data[tag] = []
                    data[tag].append(value.simple_value)
        return pd.DataFrame(data)

    out = []
    for (root, _, filenames) in os.walk(log_dir):
        for filename in filenames:
            if "events.out.tfevents" not in filename:
                continue
            file_full_path = os.path.join(root, filename)
            out.append(convert_tfevent(file_full_path))

    all_df = pd.concat(out)
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)
        
    return all_df.reset_index(drop=True)



def extract_column_from_df(df, tag_list):
    """주어진 DataFrame에서 원하는 tag들로만 이루어진 새로운 DataFrame을 만들어 반환합니다.
    Args:
    - df (pd.DataFrame): pandas DataFrame containing the log data.
    - tag_list (List[str]): Tag list for which values need to be extracted.
    
    Returns:
    - extracted_df (pd.DataFrame): pandas DataFrame containing values for the specified tag.
    """
    extracted_df = df[tag_list]
    
    return extracted_df


def df_info(df):
    """주어진 DataFrame에 대한 기본적인 정보를 출력합니다.

    Args:
    - df (pd.DataFrame): pandas DataFrame to be inspected.

    Returns:
    - None
    """
    print("DataFrame Info:")
    print("=" * 50)
    print("✲ Shape:", df.shape)
    print("✲ Columns:")
    print(df.columns)
    print("\n✲ Data Types:")
    print(df.dtypes)
    print("\n✲Non-null Value Counts:")
    print(df.count())
    
    print("\nDataFrame HEAD:")
    print("-" * 50)
    display(df.head())
    print("DataFrame TAIL:")
    print("-" * 50)
    display(df.tail())


