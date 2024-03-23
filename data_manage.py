import os
import numpy as np
import pandas as pd
from IPython.display import display
from tensorflow.python.summary.summary_iterator import summary_iterator

def load_ndarray(file_path:str):
    data = np.load(file_path)
    
    return data


def load_csv(file_path:str):
    data = pd.read_csv(file_path)
    
    return data


def load_log(log_dir, tag_filter=None, sort_by=None):
    """주어진 path에 있는 log를 읽어서, 특정 tag에 대한 value만 추출하여 DataFrame을 만들어 반환합니다.
    #! This code reference from https://gist.github.com/laszukdawid/62656cf7b34cac35b325ba21d46ecfcd 
    Args:
        log_dir: (str) tensorboard 데이터가 있는 루트 디렉토리 경로.
        tag_filter: (str or list of str) 반환할 태그 또는 태그들의 리스트.
        sort_by: (optional str) 정렬할 기준 열 이름.
    
    Returns:
        pandas.DataFrame: tag_filter에 해당하는 태그에 대한 값만 포함된 DataFrame.
    """
    def convert_tfevent(filepath, tag_filter):
        data = {}
        for e in summary_iterator(filepath):
            if len(e.summary.value) > 0:
                for value in e.summary.value:
                    tag = value.tag
                    if tag_filter is None or tag in tag_filter:
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
            out.append(convert_tfevent(file_full_path, tag_filter))

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


def revise_df(df, operation, column, value, prior_value=False):
    """DataFrame의 특정 column의 전체 값에 대해 어떤 연산을 취한다.
    """
    revised_df = df.copy()
    if operation == 'add':
        revised_df[column] += value
    elif operation == 'subtract':
        if prior_value:
            revised_df[column] = value - df[column]
        else:
            revised_df[column] -= value
    elif operation == 'multiply':
        revised_df[column] *= value
    elif operation == 'divide':
        if prior_value:
            revised_df[column] = value / df[column]
        else:
            revised_df[column] /= value
    elif operation == 'int_divide':
        if prior_value:
            revised_df[column] = value // df[column]
        else:
            revised_df[column] //= value

    return revised_df


def make_save_name(file_path):
    split_list = file_path.split("/")
    type = split_list[1]
    tag_dict = {
        "videopath": 'data',
        "rewardmethod": 'reward',
        "threshold": 'thresh',
        "epsdec": 'eps'
    }
    if type == "logs":
        tag_list = split_list[2].split("_")
        save_name = ""
        import_flag = 0
        idx = 0
        for tag in tag_list:
            name = tag
            value = tag_dict.get(tag)
            if value is not None:
                name = value
            if tag == 'importantmethod':
                import_flag += 1
                name = ""
            elif import_flag == 1:
                import_flag += 1
                name = ""
            elif idx == 0:
                idx += 1
            elif idx == 1:
                idx += 1
                save_name += name
            else:
                save_name += ("_" + name)
                
    elif type == "models":
        tag_list = split_list[2][:-4].split("_")
        save_name = ""
        import_flag = 0
        idx = 0
        for tag in tag_list:
            name = tag
            value = tag_dict.get(tag)
            if value is not None:
                name = value
            if tag == 'importantmethod':
                import_flag += 1
                name = ""
            elif import_flag == 1:
                import_flag += 1
                name = ""
            elif idx == 0:
                idx += 1
            elif idx == 1:
                idx += 1
                save_name += name
            else:
                save_name += ("_" + name)
        
    elif type == "csvs":
        save_name = split_list[2][:-4]
        
        
    return save_name


def combine_columns_into_df(column_list, df_name_list, df_list):
    """여러 DataFrame에서 특정 column의 값을 모아 하나의 DataFrame으로 만듭니다.
    
    Args:
        columns (list of str): 데이터를 모을 column들의 이름.
        column_names (list of str): 각 column에 대한 이름들.
        df_names (list of str): 각 DataFrame을 구분하기 위한 이름들.
        df_list (list of pd.DataFrame): 데이터를 가져올 여러 DataFrame들.

    Returns:
        pd.DataFrame: 모든 값이 모인 새로운 DataFrame.
    """
    combined_data = {}

    for col in column_list:
        for name, df in zip(df_name_list, df_list):
            new_name = f"{name}:{col}" 
            combined_data[new_name] = df[col]

    return pd.DataFrame(combined_data)