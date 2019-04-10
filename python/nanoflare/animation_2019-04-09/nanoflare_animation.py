import os
import h5py
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt

path = os.path.expanduser('~/documents/mas/nanoflare_videos/NanoMovie0_2000strands_94.h5')
nanoflare = h5py.File(path)['NanoMovie0_2000strands_94']

fig = plt.figure()


ims = []
for i in nanoflare[0:20]:
    im = plt.imshow(i, animated=True)
    ims.append([im])

ani = ArtistAnimation(fig, ims, interval=5, blit=True, repeat_delay=0)
ani.save("20.mp4")

plt.show()
