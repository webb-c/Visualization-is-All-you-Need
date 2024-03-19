import numpy as np
import matplotlib.pyplot as plt

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