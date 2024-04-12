import os
import numpy as np
import matplotlib.pyplot as plt


def save_plot(plot, save_path):
    """Save the given plot to a file.
    
    Args:
        plot: The plot object to save.
        filename (str): The name of the file to save the plot to.
        format (str): The format of the file to save the plot to (default: 'png').
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path+'.pdf')


def make_barcode(skip_list, num_frame, color='red', fps=30, save=False, save_path=None) :
    """Make barcode figrue. (white is skip_frame)
    Args
        skip_list (pd.DataFrame)
    """
    squares = [1] * num_frame
    idx = 0
    for s in skip_list.squeeze().values.tolist():
        for i in range(int(s)):
            if idx + i >= num_frame:
                break
            squares[idx + i] = 0
        idx += 30

    unit = num_frame / 30
    plt.figure(figsize=(unit, unit / 13))
    
    for i in range(num_frame):
        if squares[i] == 1:
            plt.bar(i, 1, color=color)

    plt.axis([0, num_frame, 0, 1])

    plt.xticks([])
    plt.yticks([])
    if save:
        save_plot(plt, save_path)

    plt.show()


def plot_dataframe(df, xlabel='step', ylabel='value', title=None, color=None, figsize=(8, 6), save_path=None, save=False, ylim=None, legend="upper right", inter=True, alpha=1):
    """Plot each column of the given DataFrame
    
    Args:
        df (pd.DataFrame)
        xlabel(str, optional): plot xlabel
        ylabel (str, optional): plot ylabel
        title (str, optional): plot title
        save (bool, optional): save option
    """
    plt.figure(figsize=figsize)
    is_one = 0
    
    for i, col in enumerate(df.columns):
        is_one += 1
        if color:
            if inter:
                # 0 값을 NaN으로 대체한 후에 보간
                interpolated_data = df[col].replace(0, np.nan).interpolate()
                plt.plot(df.index, interpolated_data, label=col, color=color[i % len(color)], alpha=alpha)
            else:
                plt.plot(df.index, df[col], label=col, color=color[i % len(color)], alpha=alpha)
        else:
            if inter:
                # 0 값을 NaN으로 대체한 후에 보간
                interpolated_data = df[col].replace(0, np.nan).interpolate()
                plt.plot(df.index, interpolated_data, label=col, alpha=alpha)
            else:
                plt.plot(df.index, df[col], label=col, alpha=alpha)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if ylim is not None:
        plt.ylim(ylim)
    if is_one > 1:
        plt.legend(loc=legend)
    if title is not None:
        plt.title(title)
    
    plt.tight_layout()
    
    if save:
        save_plot(plt, save_path)

    plt.show()


def plot_dataframe_each_plot(df, xlabel='step', ylabel='value', color=None, figsize=(10, 4), title=None, save_path=None, save=False, ylim=None, inter=True):
    """Plot each column of the given DataFrame individual graph.

    Args:
        df (pd.DataFrame): The DataFrame to plot.
        xlabel (str, optional): The x-axis label.
        ylabel (str, optional): The y-axis label.
        title (str, optional): The plot title.
        save_path (str, optional): The path to save the plot.
        save (bool, optional): Whether to save the plot.
    """
    num_cols = len(df.columns)
    unit = int(num_cols**0.5) + 1
    fig, axes = plt.subplots(nrows=unit, ncols=unit, sharex=True, figsize=(figsize[0]*unit, figsize[1]*unit))

    for i, col in enumerate(df.columns):
        row_idx = i // unit
        col_idx = i % (unit)
        
        if color:
            if inter:
                axes[row_idx, col_idx].plot(df.index, df[col].interpolate(), label=col, color=color)
            else:
                axes[row_idx, col_idx].plot(df.index, df[col], label=col, color=color)
        else:
            if inter:
                axes[row_idx, col_idx].plot(df.index, df[col].interpolate(), label=col)
            else:
                axes[row_idx, col_idx].plot(df.index, df[col], label=col)
        
        axes[row_idx, col_idx].set_xlabel(xlabel)
        axes[row_idx, col_idx].set_ylabel(ylabel)
        if title is not None:
            axes[row_idx, col_idx].set_title(f"{title} ({col})")
    
    for ax in axes.ravel():
        if not ax.lines:
            fig.delaxes(ax)
    
    plt.tight_layout()
    if save:
        save_path += "_Each"
        save_plot(plt, save_path)

    plt.show()



def plot_heatmap(data, name=None, show_minmax=False):
    plt.figure(figsize=(12, 10)) 
    plt.imshow(data, cmap='RdYlGn', interpolation='nearest')
    cbar = plt.colorbar(shrink=0.5)  
    if show_minmax:
        max_val = np.max(data)
        min_val = np.min(data)
        max_indices = np.where(data == max_val)
        min_indices = np.where(data == min_val)
        
        for i, j in zip(*max_indices):
            plt.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center', color='black', fontsize=8)  
        for i, j in zip(*min_indices):
            plt.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center', color='black', fontsize=8)  
    
    if name is not None:
        plt.title(name)
        
    plt.show()
