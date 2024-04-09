import pandas as pd
from collections import Counter
import data_manage as dm
import visualization as vis


def make_save_path(file_path, type):
    ROOT = 'result/'
    save_dict = {
        "reward": "LRLO/RL/reward/",
        "fration": "LRLO/RL/fration/",
        "F1": "LRLO/RL/F1/",
        "send": "LRLO/RL/network/send/",
        "barcode": "LRLO/RL/network/barcode/",
        "physical": "LRLO/Network/backlog/physical/",
        "virtual": "LRLO/Network/backlog/virtual/",
        "latency": "LRLO/Network/latency/",
        "path": "LRLO/Network/path/"
    }
    
    name = dm.make_save_name(file_path)
    path = ROOT + save_dict[type] + name
    return path


def parse_omnet_csv(df, type=None, node_list=None):
    """omnet: 옴넷 extract 데이터를 잘 다룰 수 있도록 형태를 변경"""
    filtered_df = df[df['type'] == 'vector']
    extracted_df = dm.extract_column_from_df(filtered_df, ["name", "vecvalue", "vectime"])
    
    if type == 'backlog' or type == 'latency':
        if type == 'backlog':
            extracted_df['name'] = extracted_df['name'].str[3:]
        rows = []
        for idx, row in extracted_df.iterrows():
            name = row['name']
            vecvalues = row['vecvalue'].split()
            vectimes = row['vectime'].split()
            for val, time in zip(vecvalues, vectimes):
                rows.append([name, float(time), float(val)])  # 시간과 값 모두 float 형태로 저장

        new_df = pd.DataFrame(rows, columns=['name', 'vectime', 'vecvalue'])
        new_df = new_df.pivot_table(index='vectime', columns='name', values='vecvalue')
        
        # dm.df_info(new_df)
        if type == 'latency' and node_list:
            new_df = dm.extract_column_from_df(new_df, node_list)

    return new_df


def get_backlog_sum(df):
    new_data = {}

    for column in df.columns:
        prefix = column.split(': ')[0]
        if prefix not in new_data:
            new_data[prefix] = df[column].copy()
        else:
            new_data[prefix] = new_data[prefix].fillna(0) + df[column].fillna(0)

    new_df = pd.DataFrame(new_data, index=df.index)
    new_df = new_df.rename(columns={'node physical backlog': 'physical backlog', 'node virtual backlog': 'virtual backlog'})
    
    new_df = new_df[['virtual backlog', 'physical backlog']]
    return new_df


def get_backlog_avg(df):
    new_data = {}
    prefix_count = {}
    
    for column in df.columns:
        prefix = column.split(': ')[0]
        if prefix not in new_data:
            new_data[prefix] = df[column].copy()
            prefix_count[prefix] = 1
        else:
            new_data[prefix] = new_data[prefix].fillna(0) + df[column].fillna(0)
            prefix_count[prefix] += 1

    for key, value in new_data.items():
        new_data[key] /= prefix_count[key]
    
    new_df = pd.DataFrame(new_data, index=df.index)
    new_df = new_df.rename(columns={'node physical backlog': 'physical backlog', 'node virtual backlog': 'virtual backlog'})
    
    new_df = new_df[['virtual backlog', 'physical backlog']]
    return new_df


def get_latency_avg(df):
    new_data = {}

    for column in df.columns:
        prefix = column.split(': ')[0]
        if prefix not in new_data:
            new_data[prefix] = df[column].copy()
        else:
            new_data[prefix] = new_data[prefix].fillna(0) + df[column].fillna(0)

    for key, value in new_data.items():
        new_data[key] /= len(df.columns)

    new_df = pd.DataFrame(new_data)

    return new_df


def get_group_backlog_df(df):
    prefix_list = []
    new_data_list = {}

    for column in df.columns:
        prefix = column.split(': ')[0]
        new_column = column.split(': ')[1]
        if prefix not in prefix_list:
            prefix_list.append(prefix)
            new_data_list[prefix] = {}
            new_data_list[prefix][new_column] = df[column].copy()
        else:
            new_data_list[prefix][new_column] = df[column].copy()

    df_list = []
    for data in new_data_list.values():
        new_df = pd.DataFrame(data)
        df_list.append(new_df)

    return df_list

def plot_reward(df, title=None, save=False, save_path=None):
    """train_log: reward plot"""
    new_df = dm.extract_column_from_df(df, ["Reward/all"]).rename(columns={"Reward/all":"reward"})
    vis.plot_dataframe(new_df, ylabel="reward", title=title, save=save, save_path=save_path, figsize=(4, 3))
    

def plot_comparison_reward(df_list, column_list, title=None, save=False, save_path=None):
    """train_log list: plot reward in same frame"""
    df_list = [ df.rename(columns={"Reward/all": "reward"}) for df in df_list ]
    combine_df = dm.combine_columns_into_df(["reward"], column_list, df_list)
    vis.plot_dataframe(combine_df, ylabel="reward", title=title, save=save, save_path=save_path, figsize=(4, 3), legend="lower right")


def plot_frame_barcode(df, color="red", save=False, save_path=None):
    extracted_df =dm.extract_column_from_df(df, ["Network/send_a(t)"])
    action_df = dm.revise_df(extracted_df, 'subtract', column="Network/send_a(t)", value=5, prior_value=True)
    vis.make_barcode(action_df, len(action_df)*6, color=color, save=save, save_path=save_path)
    

def plot_send(df, title=None, save=False, save_path=None, ylim=(-1, 31)):
    """test_log: send frame을 시간별 변화를 그림"""
    new_df = dm.extract_column_from_df(df, ["Network/send_a(t)"]).rename(columns={"Network/send_a(t)":"send"})
    vis.plot_dataframe(new_df, ylabel="frame", title=title, save=save, save_path=save_path, figsize= (6, 3), color=["royalblue"], ylim=ylim)


def plot_fraction(df, title=None, save=False, save_path=None, ylim=(-0.05, 1.05)):
    """test_log: send frame을 시간별 변화를 그림"""
    new_df = dm.extract_column_from_df(df, ["Network/send_a(t)"]).rename(columns={"Network/send_a(t)":"send"})
    new_df = dm.revise_df(new_df, "divide", "send", 30)
    vis.plot_dataframe(new_df, ylabel="fraction", title=title, save=save, save_path=save_path, figsize= (6, 3), color=["royalblue"], ylim=ylim)


def plot_send_and_guide(df, title=None, save=False, save_path=None, ylim=(-1, 31)):
    """test_log: send frame, guided frame num 시간별 변화를 그림"""
    new_df = dm.extract_column_from_df(df, ["Network/send_a(t)", "Network/target_A(t)"]).rename(columns={"Network/send_a(t)":"send", "Network/target_A(t)":"guide"})
    vis.plot_dataframe(new_df, ylabel="frame", title=title, save=save, save_path=save_path, figsize= (6, 3), color=["tomato", "royalblue"], ylim=ylim)


def plot_send_and_guide_as_fraction(df, title=None, save=False, save_path=None, ylim=(-0.05, 1.05)):
    """test_log: send frame, guided frame num 시간별 변화를 그림"""
    new_df = dm.extract_column_from_df(df, ["Network/send_a(t)", "Network/target_A(t)"]).rename(columns={"Network/send_a(t)":"send", "Network/target_A(t)":"guide"})
    new_df = dm.revise_df(new_df, "divide", "send", 30)
    new_df = dm.revise_df(new_df, "divide", "guide", 30)
    vis.plot_dataframe(new_df, ylabel="fraction", title=title, save=save, save_path=save_path, figsize= (6, 3), color=["tomato", "royalblue"], ylim=ylim)


def plot_comparison_send(df_list, column_list, title=None, save=False, save_path=None, ylim=(-1, 31)):
    """train_log list: plot reward in same frame"""
    df_list = [ df.rename(columns={"Network/send_a(t)": "send"}) for df in df_list ]
    combine_df = dm.combine_columns_into_df(["send"], column_list, df_list)
    vis.plot_dataframe(combine_df, ylabel="frame", title=title, save=save, save_path=save_path, figsize= (6, 3), color=["mediumseagreen", "tomato"], ylim=ylim)
    
    
def plot_comparison_fraction(df_list, column_list, title=None, save=False, save_path=None, ylim=(-0.05, 1.05)):
    """train_log list: plot reward in same frame"""
    df_list = [ df.rename(columns={"Network/send_a(t)": "send"}) for df in df_list ]
    df_list = [ dm.revise_df(df, "divide", "send", 30) for df in df_list ]
    combine_df = dm.combine_columns_into_df(["send"], column_list, df_list)
    vis.plot_dataframe(combine_df, ylabel="fraction", title=title, save=save, save_path=save_path, figsize= (6, 3), color=["mediumseagreen", "tomato"], ylim=ylim, smoothing_window=smoothing_window)


def plot_diff(df, title=None, save=False, save_path=None, ylim=None):
    """test_log: send frame, guided frame num 차이 시간별 변화를 그림"""
    new_df = dm.extract_column_from_df(df, ["Network/Diff"]).rename(columns={"Network/Diff":"diff"})
    vis.plot_dataframe(new_df, ylabel="frame", title=title, save=save, save_path=save_path, figsize= (6, 3), color="royalblue")


def plot_comparison_diff(df_list, column_list, title=None, save=False, save_path=None, ylim=None):
    """test_log: send frame, guided frame num 차이 시간별 변화를 그림"""
    df_list = [ df.rename( columns={"Network/Diff":"diff"}) for df in df_list ]
    combine_df = dm.combine_columns_into_df(["diff"], column_list, df_list)
    vis.plot_dataframe(combine_df, ylabel="frame", title=title, save=save, save_path=save_path, figsize= (6, 3), color=["mediumseagreen", "tomato"], ylim=ylim)


def plot_each_backlog(backlog_df, save=False, save_path=None):
    """omnet: 각 virtusl node OR virtual link OR physical node의 backlog 그래프들을 각각 따로 그림"""
    save_path_list = [None, None, None]
    if save_path is not None:
        save_path_list[0] = save_path + "_physical"
        save_path_list[0] = save_path + "_virtual_link"
        save_path_list[0] = save_path + "_virtual_node"
    
    backlog_link_df, backlog_physical_df, backlog_node_df = get_group_backlog_df(backlog_df)
    vis.plot_dataframe_each_plot(backlog_physical_df, title="physical backlog", figsize=(5, 3), save=save, save_path=save_path)
    vis.plot_dataframe_each_plot(backlog_link_df, title="virtual link backlog", figsize=(5, 3), save=save, save_path=save_path)
    vis.plot_dataframe_each_plot(backlog_node_df, title="virtual node backlog", figsize=(5, 3), save=save, save_path=save_path)


def plot_backlog(df, save=False, save_path=[None, None]):
    """omnet: virtual backlog의 합 / physical node의 합 그래프를 따로 그림"""
    virtual_df = dm.extract_column_from_df(df, ["virtual backlog"])
    physical_df = dm.extract_column_from_df(df, ["physical backlog"])
    vis.plot_dataframe(virtual_df, xlabel="time (s)", ylabel="backlog (B)", title="Virtual Backlog", save=save, save_path=save_path[0], figsize=(7, 4))
    vis.plot_dataframe(physical_df, xlabel="time (s)", ylabel="backlog (B)", title="Physical Backlog", save=save, save_path=save_path[1], figsize=(7, 4))


def plot_comparison_backlog(df_list, column_list=["masking", "unmasking"], save=False, save_path=[None, None]):
    """omnet: virtual backlog의 합 / physical node의 합 그래프를 따로, 여러개의 그래프에 대해 하나의 plot에 비교해서 그림"""
    df_list = [ df.rename(columns={"virtual backlog": "virtual", "physical backlog": "physical"}) for df in df_list ]
    virtual_df = dm.combine_columns_into_df(["virtual"], column_list, df_list)
    physical_df = dm.combine_columns_into_df(["physical"], column_list, df_list)
    vis.plot_dataframe(virtual_df, xlabel="time (s)", ylabel="backlog (B)", title="Virtual Backlog", save=save, save_path=save_path[0], figsize=(7, 4), legend="lower right", alpha=0.5, color=["royalblue", "tomato"])
    vis.plot_dataframe(physical_df, xlabel="time (s)", ylabel="backlog (B)", title="Physical Backlog", save=save, save_path=save_path[0], figsize=(7, 4), legend="lower right", alpha=0.5, color=["royalblue", "tomato"])


def plot_each_latency(latency_df):
    vis.plot_dataframe_each_plot(latency_df, title="latency", figsize=(7, 4), xlabel="time (s)", ylabel="time (s)", color="royalblue")
    """omnet: destination 노드별 layency 그래프를 그림"""
    
    
def plot_each_comparison_latency(df_list, column_list=["masking", "unmasking"], save=False, save_path=[None, None, None]):
    """omnet: virtual backlog의 합 / physical node의 합 그래프를 따로, 여러개의 그래프에 대해 하나의 plot에 비교해서 그림"""
    columns = df_list[0].columns
    for i, column in enumerate(columns):
        df = dm.combine_columns_into_df([column], column_list, df_list)
        vis.plot_dataframe(df, xlabel="time (s)", ylabel="time (s)", title="latency:"+column, save=save, save_path=save_path[i], figsize=(7, 4), legend="lower right", color=["royalblue", "tomato"])
        


def parse_routing_path(data):
    path_string = "." + data['vecvalue'].values[0][1:-2]
    str_paths = path_string.replace(" ", "").split("-1")[:-1]
    paths = list(map(lambda x:list(map(int, x.split(".")[1:-1])), str_paths))
    paths_list = [[[path[i], path[i+1]] for i in range(len(path) - 1)] for path in paths]

    element_counts = Counter(tuple(tuple(lst) for lst in sublist) for sublist in paths_list)

    unique_paths_list = list(element_counts.keys())  
    counts = list(element_counts.values())  

    print(len(unique_paths_list))
    for idx, p in enumerate(unique_paths_list) : 
        print("path: ", p)
        print("count: ", counts[idx])