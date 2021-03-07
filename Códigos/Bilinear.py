from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 18})

b = [1]
a = [1, 0]
w, h = signal.freqs(b,a,worN=np.linspace(1,150,150))

angles = np.unwrap(np.angle(h))

fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
axs[0].plot(w[1:], 20 * np.log10(abs(h[1:])), 'r', label = r'|$H(z)$|')
axs[0].set_xlabel('Frequency [rad/s]')
axs[0].set_ylabel('Amplitude response [dB]')

axs[1].plot(w, angles,'r-', label = r'arg$(H(z))$')
axs[1].set_ylabel('Angle (radians)')
axs[1].set_xlabel("Frequency [rad/s]")

axs[0].legend()
axs[1].legend()
axs[0].grid()
axs[1].grid()

plt.legend()
plt.savefig('./imagens/has', bbox_inches='tight')
plt.close()
plt.clf()





T=1/300
b1 = [1, 1]
a1 = [600, -600]
fs = 150/np.pi
w1, h1 = signal.freqz(b1,a1,worN =np.linspace(1,150,150),  fs=fs, whole = True)

angles1 = np.unwrap(np.angle(h1))

fig1, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
axs[0].plot(w1[1:], 20 * np.log10(abs(h1[1:])), 'g', label = r'|$H(z)$|')
axs[0].set_xlabel('Frequency [rad/s]')
axs[0].set_ylabel('Amplitude response [dB]')

axs[1].plot(w1, angles1,'g-', label = r'arg$(H(z))$')
axs[1].set_ylabel('Angle (radians)')
axs[1].set_xlabel("Frequency [rad/s]")

axs[0].legend()
axs[1].legend()
plt.savefig('./imagens/hz', bbox_inches='tight')
plt.close()
plt.clf()


bb, aa = signal.bilinear(b,a)
w2, h2 = signal.freqz(bb,aa, worN =np.linspace(1,150,150),  fs=fs, whole = True)

angles2 = np.unwrap(np.angle(h2))

fig2, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
axs[0].plot(w2[1:], 20 * np.log10(abs(h2[1:])), 'b', label = r'|$H(z)$|')
axs[0].set_xlabel('Frequency [rad/s]')
axs[0].set_ylabel('Amplitude response [dB]')

axs[1].plot(w2, angles2,'b-', label = r'arg$(H(z))$')
axs[1].set_ylabel('Angle (radians)')
axs[1].set_xlabel("Frequency [rad/s]")

axs[0].legend()
axs[1].legend()
plt.savefig('./imagens/hz2', bbox_inches='tight')
plt.close()
plt.clf()