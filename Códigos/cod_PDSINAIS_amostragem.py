import matplotlib.pyplot as plt
import numpy as np

def downsample(input, factor):
    output = []
    c=(factor-1)
    for i in input:
        c += 1
        if(c> (factor-1)):
            c=0
            output.append(i)
    return output

# Upsample the signal, inserting n-1 zeros between every element.
def upsample(input, factor):
    output = np.zeros(len(input)*factor)
    c=0
    for i in input:
        output[c*factor] = i
        c+=1

    return output

def amostragem(sig_in,td,ts):
    global seconds
    factor = int(ts/td)
    s_out=downsample(sig_in,factor)
    s_out=upsample(s_out,factor)

    return s_out

# Exemplo de amostragem e quantização
seconds = 1
samplerate = 500000
samplerate2 = 5000
td = 1/samplerate
t = np.linspace(0, seconds, seconds * samplerate)
xsig = 3*np.cos(1000 * 2 * np.pi * t) + 5*np.sin(3000 * 2 * np.pi * t) + 10*np.cos(6000 * 2 * np.pi * t)  # seno de 1Hz + 3Hz

Lsig=len(xsig)

ts = 1 / samplerate2  # taxa de amostragem 50 Hz
fator = int(ts / td)
# envia o sinal a um amostrador
s_out = amostragem(xsig, td, ts)

t_out_lin = np.linspace(0, seconds, seconds * samplerate2)
s_out_lin = 3*np.cos(1000 * 2 * np.pi * t_out_lin) + 5*np.sin(3000 * 2 * np.pi * t_out_lin) + 10*np.cos(6000 * 2 * np.pi * t_out_lin)  # seno de 1Hz + 3Hz


# calcula a transformada de Fourier
Lfft = int(pow(2, np.ceil(np.log2(Lsig)+1)))
Fmax = int(1 / (2 * td))
Faxis = np.linspace(-Fmax, Fmax, int(Lfft))
Xsig = np.fft.fftshift(np.fft.fft(xsig,Lfft))
S_out = np.fft.fftshift(np.fft.fft(s_out,Lfft))

# calcula o sinal reconstruído a partir de amostragem ideal e LPF (filtro passa-baixas) ideal

BW=np.floor((Lfft/fator)/2)
H_lpf=np.zeros(Lfft)

H_lpf[int(Lfft/2 - BW):int(Lfft/2+BW-1)]=1 # LPF ideal
S_recv=fator*S_out*H_lpf # filtragem ideal
s_recv=np.real(np.fft.ifft(np.fft.fftshift(S_recv))) # domínio da freq. reconstruído
s_recv=s_recv[0:Lsig] # domínio do tempo reconstruçdo

#matplotlib engine
# traça gráfico do sinal original e do sinal amostrado nos domínios do tempo e da frequencia
plt.figure(figsize=(16, 9), )
#plt.subplots_adjust(left=0.95, bottom=0.95, right=0.96, top=0.96)

plt.plot(t,xsig[0:Lsig],'k')
plt.vlines(t,0,s_out[0:Lsig], colors='green')
plt.axis([0.005,0.008, -20, 17])
plt.xlabel('tempo,segundos')
plt.title('sinal x(t) e suas amostras uniformes')
plt.savefig("fig_0", bbox_inches='tight')
plt.show()

plt.figure(figsize=(16, 9), )
plt.plot(t_out_lin,s_out_lin,'b')
plt.axis([0.005,0.008, -20, 15])
plt.xlabel('tempo,segundos')
plt.title('sinal amostrado x[n]')
plt.savefig("fig_1", bbox_inches='tight')
plt.show()

plt.figure(figsize=(16, 9), )
plt.plot(Faxis,abs(Xsig))
plt.xlabel('frequência (Hz)')
plt.axis([-8000, 8000, 0, 3700000])
plt.title('Espectro de x(t)')
plt.savefig("fig_2", bbox_inches='tight')
plt.show()

plt.figure(figsize=(16, 9), )
plt.plot(Faxis,abs(S_out))
plt.xlabel('frequência (Hz)')
plt.title('Espectro de xT(t)')
plt.axis([-8000, 8000, 0, 3700000/fator])
plt.savefig("fig_3", bbox_inches='tight')
plt.show()

# traça gráfico do sinal reconstruído idealmente nos domínios do tempo e da frequencia
plt.figure(figsize=(16, 9), )
plt.plot(Faxis,abs(S_recv))
plt.axis([-8000, 8000, 0, 3700000])
plt.xlabel('frequência (Hz)')
plt.title('Espectro de filtragem ideal (reconstrução)')
plt.savefig("fig_4", bbox_inches='tight')
plt.show()

plt.figure(figsize=(16, 9), )
line_up, = plt.plot(t,xsig,'k-.', label = 'x(t)')
line_down, = plt.plot(t,s_recv,'b', label = 'y(t)')
plt.axis([0.005,0.008, -20, 20])
plt.xlabel('tempo,segundos')
plt.title('Sinal original versus Sinal reconstruido idealmente y(t)')
plt.legend(handles=[line_up, line_down])
plt.savefig("fig_5", bbox_inches='tight')
plt.show()