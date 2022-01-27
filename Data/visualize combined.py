import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
import pandas as pd
import folium
from matplotlib import colors as mcolors


count = 0

fig, axs = plt.subplots(4,4)
axs = axs.reshape(-1)
def rec_h5(root, k=None):
    global count, fig, ax

    print("attrs for key: ", k, dict(root.attrs.items()))
    try:
        ls = list(root.keys())
        print("keylist for root: ", root, ls)
    except:
        #print("H5: ", root)
        array = np.array(root)
        #print("numpy: ", array.shape)

        #print(np.mean(array))
        axs[count].imshow(array,cmap="gray")
        axs[count].set_title(root.name)
        print(f"-------------{root.name}------------")
        count+=1

        return

    for key in ls:

        rec_h5(root[key], key)

def do_with_member(member):
    print(member)

def do_with_items(name,object):
    print(name,object)


def actual_h5(file):
    file.visititems(do_with_items)


file_name = "pn152_combined.h5"

#with h5.File(f"C:\\Users\\valte\\Desktop\\Teoretisk Fysik\\SMHI master\\Data\\18\\00\\{file_name}","r") as f:
with h5.File(f"C:\\Users\\valte\\Desktop\\Teoretisk Fysik\\SMHI master\\Data\\{file_name}","r") as f:


    #rec_h5(f)
    actual_h5(f)
    #fig.suptitle(file_name)
    #plt.show()
