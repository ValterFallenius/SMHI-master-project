import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import os
import time as TT

_already_added = []

def do_with_items(name,object):
    #name = file name in the new hdf5 "", object discarded for now (this is the array)
    global _already_added
    _already_added.append(name)

def combine_h5(files_directory,combined_directory,name,maxN=100):
    global _already_added
    with h5.File(f'combi.h5', 'r') as check: # checks if the process for combining the files already has initiated but for some reason terminated prematurely
        check.visititems(do_with_items)
        print("number before: ", len(_already_added))
        print(_already_added)
    with h5.File(f'combi.h5', 'a') as target:

        arrays = []
        while True: #the script continues forever to allow for continuous transmission, otherwise remove.

            pathlist = list(Path(files_directory).glob('**/*.h5')) #lists all .h5 files in files_directory
            TT.sleep(5)
            if len(pathlist)==0: #asks user if you want to break when pathlist is emptied of all files (removed)
                asked=input("continue?")
                if asked=="no":
                    break

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

                    target_name = files_directory+"-"+"_".join([year,month,day,time])
                    print(name)
                    array = np.array(f[name])

                    if target_name in _already_added:
                        print("already added")

                    else:
                        print(target_name)
                        target.create_dataset(target_name,data = array,compression="gzip")
                        _already_added.append(target_name)
                os.remove(path) #REMOVES FILES

if __name__ == "__main__":

    NAME = "dataset1/data1/quality1/data" #key for wanted data
    files_directory = "data/pn157/pn157"
    combined_directory = "/data"
    combine_h5(files_directory,combined_directory,NAME,None)
