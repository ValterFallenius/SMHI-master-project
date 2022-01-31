import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio
import h5py as h5
from skimage.transform import resize
import matplotlib.animation as animation


def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

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
            #print(array.shape)
            yield array, date, title




fps = 30
nSeconds = 30
h5_name = "ppi_combined.h5"
snapshots = []
datetimes = []
titles = []
for array,date,title in h5_iterator(h5_name, fps*nSeconds):
    snapshots.append(array)
    datetimes.append(date)
    titles.append(title)


SHAPE = array.shape
snapshots = [(1-a/255)**(1./4) for a in snapshots]

print(snapshots[0].shape)

#Use base cmap to create transparent
mycmap = transparent_cmap(plt.cm.Reds)


# Import image and get x and y extents
I = imageio.imread('blank2.png')
I = resize(I, SHAPE)
p = np.asarray(I).astype('float')

h,w, c = I.shape

y, x = np.mgrid[0:h, 0:w]

#Plot image and overlay colormap
fig, ax = plt.subplots(1, 1)
ax.imshow(I)

cb = ax.contourf(x, y, snapshots[0], 15, cmap=mycmap)
plt.colorbar(cb)
plt.show()

fig,ax = plt.subplots(1,1)
a = snapshots[0]
im = ax.imshow(I)
cb = ax.contourf(x, y, 1-snapshots[0]/255, 15, cmap=mycmap)
TIT =  titles[0] + " to \n" + titles[-1]
#title = fig.suptitle(TIT)

def animate_func(i):
    global cb
    for c in cb.collections:
        c.remove()  # removes only the contours, leaves the rest intact
    cb = ax.contourf(x, y, snapshots[i], 15, cmap=mycmap)
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
