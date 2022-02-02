import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import os
from datetime import datetime, timedelta

#from grid_with_projection import GridWithProjection
def h5_writer(directory,data,dates):
    with h5.File(f'network_data.h5', 'w') as target:
        for date in dates:
            target.create_dataset(target_name,data = array,compression="gzip")

def date_assertion(dates,delta = 5):
    for date1,date2 in zip(dates[0:-2],dates[1:-1]):
        list1 = date1.split("_")
        list1 = list1[0:3] + list1[3].split(":")
        #print(date1, date2)
        y1, m1, d1, hour1, minute1 =  [int(a) for a in list1]
        datetime1 = datetime(y1, m1, d1, hour=hour1, minute=minute1)
        list2 = date2.split("_")
        list2 = list2[0:3] + list2[3].split(":")
        y2, m2, d2, hour2, minute2  = [int(a) for a in list2]

        datetime2 = datetime(y2, m2, d2, hour=hour2, minute=minute2)
        delta = datetime2-datetime1
        minutes = delta.total_seconds()/60

        #print(datetime1)
        #print(datetime2)
        #print("DELTA", minutes)
        assert int(minutes) == 5

def h5_iterator(h5_file,maxN = 100):
    """Iterates through the desired datafile and returns index, array and datetime"""

    months = {"01":"January", "02":"February", "03":"March",
    "04":"April", "05":"May", "06":"June",
    "07":"July", "08":"August", "09":"September",
    "10":"October", "11":"November", "12":"December"}
    with h5.File(h5_file,"r") as f:
        for i,(name, obj) in enumerate(f.items()):
            if maxN:
                if i>=maxN: break
            #print(name)
            type, date = name.split("-")
            y, m, d, t = date.split("_")
            #title = f"Year: {y} month: {months[m]} day: {d} time: {t}"
            array = 255-np.array(obj) #flip array


            #print(array.shape)
            yield i, array, date

def down_sampler(array, rate = 2):
    """Spatial downsampling with vertical and horizontal downsampling rate = rate."""

    n,m = array.shape
    #print("TYPE: ",array.dtype)

    n_new = n//rate
    m_new = m//rate
    array_new = np.empty((n_new,m_new), dtype = np.uint8)
    for i in range(0, n, rate):
        for j in range(0, m, rate):
            try:
                array_new[i//rate,j//rate] = array[i,j]
            except IndexError:
                pass
    return array_new

def temporal_concatenation(data,dates,concat = 9, overlap = 0):
    """Takes the spatial 2D arrays and concatenates temporal aspect to 3D-vector (T-120min, T-105min, ..., T-0min)
    concat = number of frames to encode in temporal dimension
    overlap = how many of the spatial arrays are allowed to overlap in another datasample"""
    n,x_size,y_size = data.shape
    #concecutive time
    concats = []
    conc_dates=[]
    for i in range(0,n-concat,concat-overlap):
        if i%1000==0:
            print(f"\nTemporal concatenated samples: ",i)
        temp_array = data[i:i+concat,:,:]
        temp_dates = dates[i:i+concat]
        try:
            date_assertion(temp_dates)
        except AssertionError:
            print(f"Warning, dates are not alligned! Skipping: {i}:{i+concat}")
            return None
        concats.append(temp_array)
        conc_dates.append(temp_dates)
    concats = np.array(concats)
    print("Temporal concatenated shape: ", concats.shape)
    return concats,conc_dates


def load_data(h5_path,N = 3000, concat = 13,  square = (0,480,0,480), downsampling_rate = 2, overlap = 0):
    snapshots = []
    dates = []
    for i, array,date in h5_iterator(h5_path, N):
        if i%1000==0:
            print("Loaded samples: ",i)
        snapshots.append(array)
        dates.append(date)
    print("Done loading samples! \n")

    data = np.array(snapshots)
    print("\nDatatype data: ", data.dtype)
    print("\nData shape: ", data.shape)

    downsampled = []
    x0,x1,y0,y1 = square
    print(f"\nArea of interest by index: xmin = {x0}, xmax = {x1}, ymin = {y0}, ymax = {y1}")
    x_lim = slice(x0,x1)
    y_lim = slice(y0,y1)
    print("DOWNRATE ", downsampling_rate)
    for i,array in enumerate(data):
        if i%1000==0:
            print("Downsampling samples: ",i)
        section = array[x_lim,y_lim]
        down = down_sampler(section,downsampling_rate)
        downsampled.append(down)
    print("Done downsampling! \n")
    data_downsampled = np.array(downsampled)

    print("\nDatatype downsampled: ", data_downsampled.dtype)
    print("\nDownsampled data shape: ",data_downsampled.shape)
    temp_concat, new_dates = temporal_concatenation(data_downsampled,dates,concat = concat, overlap = overlap)
    print("Done concatenating! \n")
    return temp_concat, new_dates
def y_nextframe(data,dates):
    ys = []
    for i,array in enumerate(data):
        if i==0:
            continue

        y = array

        ys.append(y)
    ys = np.array(ys)

    assert data[0:-1].shape[0] == ys.shape[0]
    return data[0:-1], ys
def generate_y(data,dates,span = 10):

    mid_x = (data.shape[2])//2
    mid_y = (data.shape[3])//2
    ys = []
    N = data.shape[0]

    for i,(array,date) in enumerate(zip(data,dates)):
        if i==0:
            continue

        y = array[:, (mid_x-span):(mid_x+span),(mid_y-span):(mid_y+span)]


        y = np.mean(y,axis=1)
        y = np.mean(y,axis=1)

        ys.append(y)
    ys = np.array(ys)

    assert data[0:-1].shape[0] == ys.shape[0]
    return data[0:-1], ys

def partition(data,y,partition = 0.8 ):
    data = np.expand_dims(data, axis=-1)
    y= np.expand_dims(y, axis=-1)
    print(data.shape)
    N = data.shape[0]
    # Split into train and validation sets using indexing to optimize memory.
    indexes = np.arange(N)
    np.random.shuffle(indexes)
    train_index = indexes[: int(partition * N)]
    test_index = indexes[int(partition * N) :]

    Xtrain = data[train_index]/255
    Xtest = data[test_index]/255
    Ytrain = y[train_index]/255
    Ytest = y[test_index]/255
    return Xtrain,Xtest,Ytrain,Ytest

if __name__=="__main__":

    data = load_data("data/pn157_combined.h5",N =500)

    mean_time = []
    for i,array in enumerate(data):
        mean_time.append(np.mean(array[0,:,:]))
        '''
        if i>35:
            plt.imshow(array[0,:,:])

            plt.title(f"shape = {array[0,:,:].shape} i = {i} and np.mean = {np.mean(array)} ")
            plt.show()'''
    plt.plot(mean_time)
    plt.title("Averaged rain over time")
    plt.show()
