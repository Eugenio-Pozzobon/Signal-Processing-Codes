from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
from time import time as tm
plt.rcParams.update({'font.size': 18})


n = np.linspace(0,63,64)
x = []
h = []

for number in n:
    x.append(15*pow(0.9, number))
    h.append(pow(0.3, number))

L = len(x)+len(h)-1
l = np.linspace(0,L-1,L)

from time import process_time
t1_start = process_time()
for i in range(1, 10000):
    y = signal.convolve(x, h)

t1_stop = process_time()

print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)

y = signal.convolve(x, h)

plt.figure(figsize = (16,8), dpi = 300)
plt.stem(n,x,'b', markerfmt='bo', label = '$x[n]$')
plt.stem(n,h,'r', markerfmt='ro', label = '$h[n]$')
plt.legend()
plt.savefig('./imagens/xh', bbox_inches='tight')
plt.close()
plt.clf()

plt.figure(figsize = (16,8), dpi = 300)
plt.stem(l, y, 'g', markerfmt='go', label = '$x[n] * h[n]$')
plt.legend()
plt.savefig('./imagens/convolve', bbox_inches='tight')
plt.close()
plt.clf()

t1_start = process_time()
for i in range(1, 10000):
    X = fft.fft(x)
    H = fft.fft(h)
    Y_2 = X*H
    y_2 = fft.ifft(Y_2)
t1_stop = process_time()
print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)

X = fft.fft(x)
H = fft.fft(h)
Y_2 = X * H
y_2 = fft.ifft(Y_2)

fig, axs = plt.subplots(1, 2, figsize=(16, 9), dpi=300)

axs[0].stem(n, abs(X), 'b', markerfmt='bo', label = '$X[K]$')
axs[1].stem(n, abs(H), 'r', markerfmt='ro', label = '$H[K]$')

axs[0].legend()
axs[1].legend()

plt.savefig('./imagens/XH_', bbox_inches='tight')
plt.close()
plt.clf()

fig, axs = plt.subplots(1, 2, figsize=(16, 9), dpi=300)

axs[0].stem(n, abs(Y_2), 'm', markerfmt='mo', label = '$Y_2[K] = X[K]\cdot H[K]$')
axs[1].stem(n, y_2, 'm', markerfmt='mo', label = '$F^{-1}(Y_2[K])$')

axs[0].legend()
axs[1].legend()

plt.savefig('./imagens/Yy', bbox_inches='tight')
plt.close()
plt.clf()

plt.figure(figsize = (16,8), dpi = 300)
plt.stem(n+0.5,y[:64],'g', markerfmt='go', label = '$y_1[n]$')
plt.stem(n,y_2,'m', markerfmt='mo', label = '$y_2[n]$')
plt.legend()
plt.savefig('./imagens/sobrepor', bbox_inches='tight')
plt.close()
plt.clf()