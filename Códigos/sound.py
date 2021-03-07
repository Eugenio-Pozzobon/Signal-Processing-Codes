#Author: Eugênio Pozzobon
#last update: 23/11/2020
#credits for Victor Kich by the code for plot FFT spectrum

#imports
import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.io.wavfile import write
from scipy.signal import square
import numpy as np
import sounddevice as sd


i = 1

#path to save images
path = 'C:/Users/eugen/OneDrive/Área de Trabalho/Comunicação de Dados/Trabalho prático 1/Relatório/imagens/'
#secondary path
freqPath = 'App/'

def dbfft(time, y, ref=1, fs=200):
    """
    Calculate spectrum in dB scale
    Args:
        x: input signal
        fs: sampling frequency
        ref: reference value used for dBFS scale. 32768 for int16 and 1 for float

    Returns:
        freq: frequency vector
        s_db: spectrum in dB scale
    """
    N = len(time)
    fs = len(time)/(time[len(time)-1]-time[1])
    win = np.hanning(N)
    x = y * win  # Take a slice and multiply by a window
    sp = np.fft.rfft(x)  # Calculate real FFT
    s_mag = np.abs(sp) * 2 / np.sum(win)  # Scale the magnitude of FFT by window and factor of 2,
    # because we are using half of FFT spectrum
    s_dbfs = 20 * np.log10(s_mag / ref)  # Convert to dBFS
    freq = np.arange((N / 2) + 1) / (float(N) / fs)  # Frequency axis

    return freq[:-1], s_dbfs

def plotgraphs(rate, data, color):
    global i
    fig = plt.figure(figsize=(12, 12))
    plt.subplots_adjust(hspace = 1)
    time = np.linspace(0, len(data) / rate, len(data))

    #plot signal
    plt.plot(time, data, color = color)
    plt.title('Signal')
    plt.ylabel('magnitude')
    plt.xlabel('time (s)')
    plt.savefig(transparent=True, fname=path + freqPath + 'Figure_'+ str(i) + '_0.png')
    plt.show()

    # plot fft spectrum
    #calculate
    td = 1/rate
    xsig = data
    Lsig = len(xsig)
    Lfft = int(pow(2, np.ceil(np.log2(Lsig))))
    Fmax = int(1 / (2 * td))
    Faxis = np.linspace(-Fmax, Fmax, int(Lfft + 1))
    Xsig = np.fft.fftshift(np.fft.fft(xsig, Lfft) / Lfft)
    #make plot
    plt.vlines(Faxis[0:int(Lfft)], abs(Xsig), np.zeros(len(Xsig)), color = color)
    plt.title('Signal FFT')
    plt.ylabel('magnitude')
    plt.xlabel('freq')
    plt.savefig(transparent=True, fname=path + freqPath + 'Figure_'+ str(i) + '_1.png')
    plt.show()

    # plot fft with dB scale
    x_axis, y_axis = dbfft(time=time, y=data, fs=rate)
    plt.plot(x_axis, (y_axis[:-1]), color = color)
    plt.title('Signal FFT (dB x freq)')
    plt.ylabel('magnitude (dB)')
    plt.xlabel('freq')
    plt.savefig(transparent=True, fname=path + freqPath + 'Figure_'+ str(i) + '_2.png')
    plt.show()

    # now with x axis in log scale
    plt.plot(x_axis, (y_axis[:-1]), color = color)
    plt.xscale('log')
    plt.title('Signal FFT (dB x freq (Log scale))')
    plt.ylabel('magnitude (dB)')
    plt.xlabel('freq')
    plt.savefig(transparent=True, fname=path + freqPath + 'Figure_'+ str(i) + '_3.png')
    plt.show()

seconds = 5  # Duration for all waves
samplerate = 512 #rate for normal waves

t = np.linspace(0, seconds, seconds*samplerate)
amplitude = 4/np.pi

#sen
freqsen = 1
senAudio = np.sin(t * 2 * np.pi * freqsen)
plotgraphs(samplerate, senAudio, color='magenta')
i+=1

#pulse
valor_pico = 1
nivel_dc = 1
frpulse = 1
pulseAudio = (square(2 * np.pi * frpulse * t, 0.5)*valor_pico + nivel_dc)/2
plotgraphs(samplerate, pulseAudio, color='green')
i+=1

#square
#square without function, using senoids sum
#fsSquare = np.linspace(1,100,100)
#squareAudio = np.linspace(0, 0, seconds*samplerate)
#for f in fsSquare:
#    squareAudio += (np.sin((2. * f - 1 )* t * np.pi * 2000))/(2. * f - 1 )
#
#squareAudio = amplitude * squareAudio

valor_pico = 1
nivel_dc = 0
frpulse = 1
squareAudio = (square(2 * np.pi * frpulse * t)*valor_pico + nivel_dc)
plotgraphs(samplerate, squareAudio, color='blue')
i+=1

#record Audio
fsrec = 44100  # Sample rate

myrecording = sd.rec(int(seconds * fsrec), samplerate=fsrec, channels=2)
sd.wait()  # Wait until recording is finished

write('record.wav', fsrec, myrecording)  # Save as WAV file

rtrec, dtrec = wav.read('record.wav')
plotgraphs(rtrec, dtrec[:,1], color='red') #dtrec have 2 channel array, take just one