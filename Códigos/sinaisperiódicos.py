import matplotlib.pyplot as plt
import numpy as np
import math

n = np.linspace(-20,20,14000)
rd = lambda j: round(j)
n2 = np.linspace(-20,20,14000)

kt = 10

func = lambda i: math.cos(math.pi*(i+kt)*(i+kt)/8) # + 2*math.pi)

x = np.vectorize(func)
N = np.vectorize(rd)

plt.scatter(N(n), x(N(n)), color = 'red')
plt.plot(n, x(n2))
plt.xlabel('n')
plt.ylabel('x(n)')
plt.show()

