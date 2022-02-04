import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import imageio
from skimage.transform import resize

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
for array,date,title in h5_iterator(f"C:\\Users\\valte\\Desktop\\Teoretisk Fysik\\SMHI master\\Network\\data\\{h5_name}", fps*nSeconds):
    print(array,date,title)
    snapshots.append(array)
    datetimes.append(date)
    titles.append(title)
SHAPE = array.shape
N = len(snapshots)
background = imageio.imread("coords2.jpg")
background = resize(background, SHAPE)
background = np.mean(background,axis=2)
print(np.min(background),np.max(background))
print("SHAPE", background.shape)
plt.imshow((255*background+array)/2,cmap="gray")
plt.show()
'''AVG = np.zeros(SHAPE)
diff = np.zeros(SHAPE)
for a,b in zip(snapshots[0:-2],snapshots[1:-1]):
    diff += a-b
    AVG += a
diff /= N-1
AVG /= N-1
diff_max = np.max(diff)/2

diff[diff<diff_max] = 0

AVG_max = np.max(AVG)/2

#AVG[AVG<AVG_max] = 0.1
#AVG[AVG>AVG_max] = 1

plt.imshow(AVG,cmap="gray")
plt.show()
shape = (1124,906)
for i, a in enumerate(snapshots):
    #array = np.divide(a,AVG)

    #array = np.minimum(a,255)

    a[AVG<AVG_max] = a[AVG<AVG_max] + 255-AVG[AVG<AVG_max]
    a = np.minimum(a,255)
    snapshots[i] = a'''
plt.imshow(snapshots[40])

plt.title("Radar composite precipitation field")
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
