#!/usr/bin/env python3


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.size"] = 11.*.8
mpl.rcParams["axes.unicode_minus"] = False

# set up figure size
fig, ax = plt.subplots(figsize=(2.8, 2.0))

# do some plotting here
x = np.linspace(-2, 2, 1e3)
ax.plot(x, x**2)

ax.set_title("This is a text")
# save to file
plt.savefig("example.pgf")

plt.show()
