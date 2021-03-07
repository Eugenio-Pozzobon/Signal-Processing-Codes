import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile as wav
from scipy.io.wavfile import write

#--------
# record Audio

def plot_audio():
    import sounddevice as sd

    # Exemplo de amostragem e quantização
    seconds = 1
    fsrec = 44100  # Sample rate
    samplerate = fsrec
    td = 1/samplerate
    t = np.linspace(0, seconds, seconds * samplerate)
    myrecording = sd.rec(int(seconds * fsrec), samplerate=fsrec, channels=2)
    sd.wait()  # Wait until recording is finished
    write('record.wav', fsrec, myrecording)  # Save as WAV file
    rtrec, dtrec = wav.read('record.wav')
    xsig = dtrec[:, 1]
    Lsig=len(xsig)

    ts=1/fsrec # taxa de amostragem 50 Hz
    fator=ts/td
    # envia o sinal a um amostrador
    s_out = xsig #aamostragem(xsig,td,ts)

    # calcula a transformada de Fourier
    Lfft = int(pow(2, np.ceil(np.log2(Lsig)+1)))
    Fmax = int(1 / (2 * td))
    Faxis = np.linspace(-Fmax, Fmax, int(Lfft))
    Xsig = np.fft.fftshift(np.fft.fft(xsig,Lfft))
    S_out = np.fft.fftshift(np.fft.fft(s_out,Lfft))

    # calcula o sinal reconstruído a partir de amostragem ideal e LPF (filtro passa-baixas) ideal
    BW=10000#floor((Lfft/Nfactor)/2)
    H_lpf=np.zeros(Lfft)

    H_lpf[int(Lfft/2 - BW):int(Lfft/2+BW-1)]=1 # LPF ideal
    S_recv=fator*S_out*H_lpf # filtragem ideal
    s_recv=np.real(np.fft.ifft(np.fft.fftshift(S_recv))) # domínio da freq. reconstruído
    s_recv=s_recv[0:Lsig] # domínio do tempo reconstruçdo

    #matplotlib engine
    # traça gráfico do sinal original e do sinal amostrado nos domínios do tempo e da frequencia
    plt.figure(figsize=(10, 10))
    plt.plot(t,xsig[0:Lsig],'k')
    plt.vlines(t,0,s_out[0:Lsig], colors='green')
    plt.xlabel('tempo,segundos')
    plt.title('sinal g(t) e suas amostras uniformes')
    plt.show()

    plt.plot(Faxis,abs(Xsig))
    plt.xlabel('frequência (Hz)')
    plt.axis([-150, 150, 0, 300/fator])
    plt.title('Espectro de g(t)')
    plt.show()

    plt.plot(Faxis,abs(S_out))
    plt.xlabel('frequência (Hz)')
    plt.title('Espectro de g_T(t)')
    plt.axis([-150, 150, 0, 300/fator])
    plt.show()

    # traça gráfico do sinal reconstruído idealmente nos domínios do tempo e da frequencia
    plt.plot(Faxis,abs(S_recv))
    plt.axis([-150, 150, 0, 300])
    plt.xlabel('frequência (Hz)')
    plt.title('Espectro de filtragem ideal (reconstrucao)')
    plt.show()

    plt.plot(t,xsig,'k-.')
    plt.plot(t,s_recv,'b')
    plt.xlabel('tempo,segundos')
    plt.title('Sinal original versus Sinal reconstruido idealmente')
    plt.show()