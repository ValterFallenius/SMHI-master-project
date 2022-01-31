import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from skimage.filters import threshold_otsu,threshold_mean,threshold_li,threshold_triangle,threshold_isodata,threshold_yen
import skimage.morphology

def do_with_member(member):
    print(member)

def do_with_items(name,object):
    yield name,object



def h5_iterator(h5_file,maxN = 100):
    months = {"01":"January", "02":"February", "03":"March",
    "04":"April", "05":"May", "06":"June",
    "07":"July", "08":"August", "09":"September",
    "10":"October", "11":"November", "12":"December"}
    with h5.File(h5_file,"r") as f:
        for i,(name, obj) in enumerate(f.items()):
            if i%100==0:
                print(i)
            if maxN:
                if i>=maxN: break
            type, date = name.split("-")
            y, m, d, t = date.split("_")
            title = f"Year: {y} month: {months[m]} day: {d} time: {t}"
            array = np.array(obj)

            yield array, date, title




fps = 30
nSeconds = 5
h5_name = "ppi_combined.h5"
snapshots = []
datetimes = []
titles = []
for array,date,title in h5_iterator(h5_name, fps*nSeconds):
    print(array,date,title)
    snapshots.append(array)
    datetimes.append(date)
    titles.append(title)
SHAPE = array.shape
N = len(snapshots)
AVG = np.zeros(SHAPE)

ALL_snapshots = []
ALL_datetimes = []
ALL_titles = []
AVG2 = np.zeros(SHAPE)
for array,date,title in h5_iterator(h5_name, 4000):
    #print(array,date,title)
    image_background = skimage.morphology.erosion(array, skimage.morphology.disk(2)) # erosion increase background but not black letter
    image_filter=array-image_background #remove background
    image_filter = image_filter-np.min(image_filter) #normalize min=0
    AVG += array
    AVG2 +=  image_filter
    ALL_snapshots.append(array)
    ALL_datetimes.append(date)
    ALL_titles.append(title)
N = len(ALL_snapshots)
AVG /= N
AVG2 /= N
AVG2 = np.multiply(np.array(AVG2>70),AVG2)
fig,ax = plt.subplots(2,1)
ax[0].imshow(AVG)
ax[0].set_title("AVG")
ax[1].imshow(AVG2)
ax[1].set_title("AVG erosion and normalized")
plt.show()

ALL_snapshots = [np.maximum(a-AVG,0) for a in snapshots]
plt.imshow(ALL_snapshots[0])
plt.show()

# First set up the figure, the axis, and the plot element we want to animate
fig,ax = plt.subplots(1,1)
snapshots = ALL_snapshots
a = snapshots[0]
im = ax.imshow(a,cmap="hot")
TIT =  titles[0] + " to \n" + titles[-1]
#title = fig.suptitle(TIT)

def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(snapshots[i])
    ax.set_title(titles[i])
    return [im]

anim = animation.FuncAnimation(
                               fig,
                               animate_func,
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

print('Done!')

# plt.show()  # Not required, it seems!
