import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

def h5_iterator(directory,name,maxN = 100):
    pathlist = Path(directory).glob('**/*.h5')

    months = {"01":"January", "02":"February", "03":"March",
    "04":"April", "05":"May", "06":"June",
    "07":"July", "08":"August", "09":"September",
    "10":"October", "11":"November", "12":"December"}
    skipper = 1
    for i,path in enumerate(pathlist):
        if not i%skipper==0:
            continue
        if i>skipper*maxN:
            break
        # because path is object not string
        path_in_str = str(path)

        # print(path_in_str)
        with h5.File(path_in_str,"r") as f:
            print(f[name].shape, i)

            date = path_in_str.split("_")[-2]

            year = date[0:4]
            month = date[4:6]
            day = date[6:8]
            time = str(date[9:11])+":"+str(date[11:13])
            title = f"Year: {year} month: {months[month]} day: {day} time: {time}"
            print([type(x) for x in (np.array(f[name]), date, title)])
            yield np.array(f[name]), date, title




fps = 30
nSeconds = 30
NAME = "dataset1/data1/quality1/data"
snapshots = []
datetimes = []
titles = []
for array,date,title in h5_iterator("ppi",NAME, fps*nSeconds):
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
