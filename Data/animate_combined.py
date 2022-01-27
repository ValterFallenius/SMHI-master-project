import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path


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
            if i>=maxN: break
            type, date = name.split("-")
            y, m, d, t = date.split("_")
            title = f"Year: {y} month: {months[m]} day: {d} time: {t}"
            array = np.array(obj)
            print(array.shape)
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

shape = (1124,906)
# First set up the figure, the axis, and the plot element we want to animate
fig,ax = plt.subplots(1,1)

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
