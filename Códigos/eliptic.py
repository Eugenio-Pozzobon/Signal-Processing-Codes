from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

rp = 0.5 #dB,
e = np.sqrt(pow(10,(rp/10)) - 1)
A = 30 #- 20*np.log10(e)# dB
rs = A - 20*np.log10(e)# dB
Ws = 8000 #* (2 * np.pi) #[2100, 8000] #freqs 2100 * 2 *np.pi,
Wp = 2100 #* (2 * np.pi)

k = e/np.sqrt(pow(A,2) - 1)
k_ = np.sqrt(1 - pow((Wp/Ws),2))
po = (1-np.sqrt(k_))/(2*(1+np.sqrt(k_)))
p = po + 2*pow(po,5) + 15*pow(po,9) + 150*pow(po,13)

N = int((2*np.log10(4/k))/(np.log10(1/p)))

print(rs)
print((Ws/Wp))
print(N)

[b, a] = signal.ellip(N, rp, rs, Ws , 'low', analog=True)

w, h = signal.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)), color='red')
plt.title('Elliptic filter frequency response')
plt.xlabel('Frequência')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', axis='both')
plt.axvline(Ws, color='orange') # cutoff frequency
plt.axhline(-A, color='orange') # rp
plt.axhline(-rp, color='green') # rs
plt.axvline(Wp, color='green') # cutoff frequency
plt.savefig('./plot', bbox_inches='tight')
plt.show()
plt.close()

plt.semilogx(w, 20 * np.log10(abs(h)), color='red')
plt.title('Elliptic filter frequency response')
plt.xlabel('Frequência')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', axis='both')
plt.axvline(Ws, color='orange') # cutoff frequency
plt.axhline(-A, color='orange') # rp
plt.axhline(-rp, color='green') # rs
plt.axvline(Wp, color='green') # cutoff frequency
plt.axis([1000, 9000, -0.75, 0.25])
plt.savefig('./zoom', bbox_inches='tight')
plt.show()