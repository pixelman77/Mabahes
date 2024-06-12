import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 4*np.pi, 0.1)
ysi = np.sin(x)
yco = np.cos(x)

plt.plot(x, ysi, color = 'red')
plt.plot(x, yco, color = 'blue')
plt.fill_between(x, ysi, yco, hatch='///////')
plt.show()