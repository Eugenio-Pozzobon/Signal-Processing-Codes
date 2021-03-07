import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import square, sawtooth
from amostraquant import amostra_quant

SQNRvalues = []
emqvalues = []
bitsmin = 3
bitsmax = 16

#folder = '../imagens/q2/'
folder = '../imagens/q3/composto/'
#folder = '../imagens/q3/quadrada/'
#folder = '../imagens/q3/triangular/'

mf = 'go'
col = 'g'

for bits in range(bitsmin,bitsmax+1):
    # Exemplo de amostragem e quantização
    td=0.002 # intervalo entre os pontos do sinal "analógico"
    samplerate = int(1/td)
    seconds = 1
    t = np.linspace(td, seconds, seconds * samplerate)

    signal_freq = 5
    #xsig = np.sin(2 * 3 * np.pi * t)  # SEN
    xsig = np.sin(3 * 2 * np.pi * t) + np.sin(1 * 2 * np.pi * t)  # COMPOSE
    #xsig = square(2 * signal_freq * np.pi * t) # SQUARE
    #xsig = sawtooth(2 * signal_freq * np.pi * t)  # TRIANGLE

    Lsig=len(xsig)

    ts=0.02 # taxa de amostragem 50 Hz
    fator=ts/td

    plot = plt.figure(figsize=(22, 9), )
    plt.plot(t,xsig,'k', linewidth = 5)
    plt.xlabel('tempo,segundos')
    plt.axis([0,1, -2, 2])
    plt.title('sinal')
    plot.savefig(folder + "fig1_wave", bbox_inches='tight')
    plt.close()

    # amp = 2
    amp = max(xsig)
    #bits = 3  # numero de bits da conversão analógico/digital
    n=2**bits # numero de niveis

    # envia o sinal a um amostrador e quantizador
    s_out, sq_out = amostra_quant(xsig,td,ts,amp,n)

    plot2 = plt.figure(figsize=(20, 9), )
    plt.stem(t,s_out, label='amostras')
    plt.stem(t,sq_out,col, markerfmt = mf, label='amostras quantizadas')

    # ruido de quantizacao
    sr = s_out - sq_out # erro
    emq = sr.dot(np.transpose(sr))/len(sr) # erro medio quadratico
    emqvalues.append(emq)

    SQNR=20*np.log10(np.linalg.norm(s_out)/np.linalg.norm(sr)) # valor da SQNR
    SQNRvalues.append(SQNR)

    plt.plot(t,sr,'k', label='erro')
    plt.legend()
    plt.axis([0,1, -2, 2])
    plt.xlabel('tempo,segundos')
    plt.title('sinal g(t) e suas amostras uniformes e quantizadas com '+str(bits)+'bits')
    plot2.savefig(folder + "fig2_"+str(bits)+'bits', bbox_inches='tight')
    plt.close()

    plot3 = plt.figure(figsize=(20, 9), )
    plt.plot(t,xsig,'k', linewidth = 5, label='signal')
    plt.stem(t,sq_out,col, markerfmt = mf, label='amostras quantizadas')
    plt.legend()
    plt.axis([0,1, -2, 2])
    plt.xlabel('tempo,segundos')
    plt.title('sinal g(t) e suas amostras uniformes e quantizadas com '+str(bits)+'bits')
    plot3.savefig(folder + "fig3_"+str(bits)+'bits', bbox_inches='tight')
    plt.close()

xpoints = np.linspace(bitsmin, bitsmax, bitsmax-bitsmin+1)

import pandas as pd
d = {'bits': xpoints, 'emq': emqvalues, 'SQNR': SQNRvalues}
df = pd.DataFrame(data=d)
df.to_csv(folder + 'SQNR-bits.csv')

#plotSQNR = plt.figure(figsize=(16, 9), )
#plt.stem(xpoints,SQNRvalues,'r',)
#plt.plot(xpoints,SQNRvalues,'k')
#plt.xlabel('bits')
#plt.xlabel('SQNR,dB')
#plt.title('SQNR por nível de quantização')
#plotSQNR.savefig(folder + 'figSQNR', bbox_inches='tight')
#plt.close()