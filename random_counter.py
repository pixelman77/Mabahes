import numpy as np
import matplotlib.pyplot as plt

seed = 400121011
np.random.seed(seed)

x = np.arange(0, 21, 1)
y = [0] * 21
print(y)

for i in range(10000):
    randi = np.random.randint(0, 21)
    print(str(randi) + "\n")
    y[randi] += 1

print(y)
plt.bar(x, y)
plt.show()

