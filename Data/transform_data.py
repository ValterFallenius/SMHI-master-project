import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import os

def combine_h5(directory,name,maxN=100):
    pathlist = Path(directory).glob('**/*.h5')
    with h5.File(f'{directory}_combined.h5', 'w') as target:
        arrays = []

        for i,path in enumerate(pathlist):

            path_in_str = str(path)
            if maxN:
                if i>=maxN: break
            with h5.File(path_in_str,"r") as f:
                date = path_in_str.split("_")[-2]
                year = date[0:4]
                month = date[4:6]
                day = date[6:8]
                time = str(date[9:11])+":"+str(date[11:13])
                target_name = directory+"-"+"_".join([year,month,day,time])
                array = np.array(f[name])
                print(target_name)
                target.create_dataset(target_name,data = array,compression="gzip")


NAME = "dataset1/data1/quality1/data"

directory = "ppi"

combine_h5(directory,NAME,maxN=None)
