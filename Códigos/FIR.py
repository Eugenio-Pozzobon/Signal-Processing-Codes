from numpy import cos, sin, pi, absolute, arange, zeros
from scipy.signal import kaiserord, lfilter, firwin, freqz
import numpy as np
import scipy.signal as ss

#------------------------------------------------
# Create a signal
#------------------------------------------------

sample_rate = 44100
nsamples = sample_rate
t = arange(nsamples) / sample_rate

x = zeros(nsamples)
x += 0.3*cos(2*pi*500*t)
x += 0.3*cos(2*pi*6000*t)
x += 0.3*cos(2*pi*10000*t)
#------------------------------------------------
# Create a FIR filter and apply it to x.
#------------------------------------------------

# The Nyquist rate of the signal.
nyq_rate = sample_rate / 2.0

# The desired width of the transition from pass to stop,
# relative to the Nyquist rate.
width = 1000/nyq_rate

# The desired attenuation in the stop band, in dB.

ripple_db = 60.0
bandpass = 4000
riple_pass = 0.1

# The cutoff frequency of the filter.
cutoff_hz = 5000

#formula kaiser
phi_p = 1 - pow(10,(-riple_pass/20))
phi_s = pow(10,(-ripple_db/20))
omega_p = 2*np.pi* (bandpass) / (sample_rate)
omega_s =2*np.pi* (cutoff_hz) / (sample_rate)

N = int((-20*np.log10(np.sqrt(phi_p*phi_s))-13)/(14.6*(omega_s - omega_p)/(2*np.pi)))

#formula bellanger
N = int(-2*np.log10(10*phi_p*phi_s)/(3*(omega_s - omega_p)/(2*np.pi)) - 1)

#formula Hermann

a_1 = 0.005309
a_2 = 0.07114
a_3 = -0.4761
a_4 = 0.00266
a_5 = 0.5941
a_6 = 0.4278
b_1 = 11.01217
b_2 = 0.51244

D=(a_1*pow(np.log10(phi_p),2)+ a_2*np.log10(phi_p)+a_3)*np.log10(phi_s) - (a_4*pow(np.log10(phi_p),2)+a_5*np.log10(phi_p)+a_6)
F=b_1+b_2*(np.log10(phi_p) - np.log10(phi_s))
N = int((D - F*pow(((omega_s - omega_p)/(2*np.pi)),2))/((omega_s - omega_p)/(2*np.pi)))

#Compute the order and Kaiser parameter for the FIR filter.
#python scipy method
N, beta = kaiserord(ripple_db, width)
print(N)

# Use firwin with a Kaiser window to create a lowpass FIR filter.
taps = firwin(N, cutoff_hz-500, fs = sample_rate, pass_zero='lowpass')

# Use lfilter to filter x with the FIR filter.
filtered_x = lfilter(taps, 1.0, x)





from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show
from pylab import savefig, semilogy, semilogx, axvline, axhline, legend
import pylab as plt
#------------------------------------------------
# Plot the FIR filter coefficients.
#------------------------------------------------

figure(1, figsize=(16,9), dpi = 300)
plot(taps, 'bo-', linewidth=2)
title('Filter Coefficients (%d taps)' % N)
savefig('./imagens/design', bbox_inches='tight')

#------------------------------------------------
# Plot the magnitude response of the filter.
#------------------------------------------------


clf()
#figure(2, figsize=(16,9), dpi = 300)
fig, ax1 = plt.subplots(figsize=(16,9), dpi = 300, constrained_layout=True)
w, h = freqz(taps)

ax1.plot((w/(pi))*nyq_rate, 20*np.log10(np.abs(h)), linewidth=2)
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Gain')
ax1.set_title('Filter Response')
ylim(-80, 10)
ax1.grid(True)

ax1.axvline(cutoff_hz,color='r', linewidth=1) # cutoff frequency
ax1.axhline(-riple_pass,color='r', linewidth=1) # rp
ax1.axhline(riple_pass,color='r', linewidth=1) # rp
ax1.axhline(-ripple_db,color='r', linewidth=1) # rs
ax1.axvline(bandpass,color='r', linewidth=1) # cutoff frequency

ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h))
ax2.plot((w/(pi))*nyq_rate, angles, color='orange')
ax2.set_ylabel('Angle (radians)', color='orange')
ax2.axis('tight')

# Upper inset plot.
ax3 = axes([0.42, 0.6, .45, .25])
plot((w/pi)*nyq_rate, 20*np.log10(absolute(h)), linewidth=2)
xlim(0,bandpass+1500)
ylim(-0.11, 0.11)
grid(True)
axvline(cutoff_hz, color='red', linewidth=1) # cutoff frequency
axhline(-riple_pass, color='red', linewidth=1) # rp
axhline(riple_pass, color='red', linewidth=1) # rp
axhline(-ripple_db, color='red', linewidth=1) # rs
axvline(bandpass, color='red', linewidth=1) # cutoff frequency

savefig('./imagens/filter', bbox_inches='tight')

#------------------------------------------------
# Plot the original and filtered signals.
#------------------------------------------------
clf()
# The phase delay of the filtered signal.
delay = 0.5 * (N-1) / sample_rate

figure(3, figsize=(16,9), dpi = 300)
# Plot the original signal.
plot(t, x, 'r', label='original signal x(t)')
# Plot the filtered signal, shifted to compensate for the phase delay.
#plot(t-delay, filtered_x, 'r', label='filtered signal y(t)')
# Plot just the "good" part of the filtered signal.  The first N-1
# samples are "corrupted" by the initial conditions.
plot(t[N-1:]-delay, filtered_x[N-1:], 'k', linewidth=4, label='filtered signal y(t)')
xlim(0.1, 0.11)

legend()

xlabel('t')

savefig('./imagens/result', bbox_inches='tight')