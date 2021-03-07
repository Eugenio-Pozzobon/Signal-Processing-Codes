import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import square, sawtooth
from scipy.io.wavfile import write
import sounddevice as sd

'''
Autor: Eugênio Pozzobon
Data: 13/01/2020
'''

# define constantes para o programa
A = 87.6
bits = 13
amp = (pow(2,bits-1)) #variável que define a amplitude máxima de um sinal inteiro para que possa ser convertido na escala de trabalho das operações de codificação da lei A

def quantizacao(sinal,amp,n):
    '''
    Realiza a quantização de um sinal
    :param sinal: sinal a quantizar
    :param amp: amplitude do sinal
    :param n: número de níveis de quantização
    :return: sinal quantizado
    '''
    # quantizacao
    Delta=2*amp/n  # quantizacao uniforme(Delta)
    x = sinal + np.array(amp)  # somar nivel CC igual a amp
    q = np.round(x/Delta)  # dividir em intervalos iguais a D
    x = Delta/2 + Delta*q - np.array(amp)  # quantizar e remover nivel CC
    return x

def conversao_ad(xsig):
    '''
    Realiza a quantização de um sinal, convertendo-o em um sinal digital
    :param xsig: sinal
    :return:
    '''

    amp = max(xsig)
    n=2**bits # numero de niveis de quantização

    # envia o sinal a um amostrador e quantizador
    Delta=2*amp/n  # quantizacao uniforme(Delta)
    x = xsig + np.array(amp)  # somar nivel CC igual a amp
    q = np.round(x/Delta)  # dividir em intervalos iguais a D
    x = Delta/2 + Delta*q - np.array(amp)  # quantizar e remover nivel CC

    return x

def encode_a_law(signal):
    '''
    retorna o sinal com a lei A
    :param signal: sinal
    :return: sinal codificado
    '''
    #signal = signal.astype(np.int16)
    quantized_signal = np.zeros(len(signal))
    signal_b = signal / amp
    i=0
    for sig in signal_b:
        if (sig < (1/A)) & (sig >= 0):
            sgn = 1
            quantized_signal[i] = sgn*( A*sig) / (1+np.log(A))
            #quantized_signal[i] = sgn * 1 / ((1 + np.log(A)) * sig / A)
        elif (sig > -(1/A)) & (sig < 0):
            sgn = -1
            sig = -sig
            quantized_signal[i] = sgn*( A*sig) / (1+np.log(A))
            #quantized_signal[i] = sgn * 1 / ((1 + np.log(A)) * sig / A)
        elif (sig >= (1/A)) & (sig <= 1):
            sgn = 1
            quantized_signal[i] = sgn*(1+np.log(A*sig)) / (1+np.log(A))
            #quantized_signal[i] = sgn * 1 / np.exp(sig * (1 + np.log(A)) - 1) / (A + A * np.log(A))
        elif (sig <= -(1/A)) & (sig >= -1):
            sgn = -1
            sig = -sig
            quantized_signal[i] = sgn * (1 + np.log(A*sig ))/ (1 + np.log(A))
            #quantized_signal[i] = sgn * 1 / np.exp(sig * (1 + np.log(A)) - 1) / (A + A * np.log(A))

        i += 1

    quantized_signal = quantized_signal* amp
    return quantized_signal

def decode_a_law(signal):
    '''
    desfaz a lei A
    :param signal: sinal codificado
    :return: sinal decodificado
    '''
    quantized_signal = np.zeros(len(signal))
    i=0

    signal_b = signal / amp
    for sig in signal_b:
        if (sig <= (1/(1+np.log(A)))) & (sig >= 0):
            sgn = 1
            quantized_signal[i] = sgn*( (1+np.log(A))*sig / A)

        elif (sig >= -(1/(1+np.log(A)))) & (sig < 0):
            sgn = -1
            sig = -sig
            quantized_signal[i] = sgn*( (1+np.log(A))*sig / A)

        elif (sig > (1/(1+np.log(A)))) & (sig <= 1):
            sgn = 1
            quantized_signal[i] = sgn* np.exp(sig*(1+np.log(A))-1) / (A)

        elif (sig < -(1/(1+np.log(A)))) & (sig >= -1):
            sgn = -1
            sig = -sig
            quantized_signal[i] = sgn* np.exp(sig*(1+np.log(A))-1) / (A)
        i += 1

    quantized_signal = quantized_signal * amp
    return quantized_signal

def compress(signal):
    '''
    realiza a compressão do sinal de acordo com a lei A
    :param signal: sinal
    :return: sinal comprimido
    '''
    signal = signal.astype(np.int16)
    new_sig = np.zeros(len(signal)).astype(np.int16)
    i = 0
    for sig in signal:
        if sig > 0:
            if   (sig >= int(0b0000000000001)) & (sig <= int(0b0000000011111)):
                new_sig[i] = int(0b0000000) + int((sig >> 1) & 0b1111)
            elif (sig >= int(0b0000000010000)) & (sig <= int(0b0000000111111)):
                new_sig[i] = int(0b0010000) + int((sig >> 1) & 0b1111)
            elif (sig >= int(0b0000000100000)) & (sig <= int(0b0000001111111)):
                new_sig[i] = int(0b0100000) + int((sig >> 2) & 0b1111)
            elif (sig >= int(0b0000001000000)) & (sig <= int(0b0000011111111)):
                new_sig[i] = int(0b0110000) + int((sig >> 3) & 0b1111)
            elif (sig >= int(0b0000010000000)) & (sig <= int(0b0000111111111)):
                new_sig[i] = int(0b1000000) + int((sig >> 4) & 0b1111)
            elif (sig >= int(0b0000100000000)) & (sig <= int(0b0001111111111)):
                new_sig[i] = int(0b1010000) + int((sig >> 5) & 0b1111)
            elif (sig >= int(0b0001000000000)) & (sig <= int(0b0011111111111)):
                new_sig[i] = int(0b1100000) + int((sig >> 6) & 0b1111)
            elif (sig >= int(0b0010000000000)) & (sig <= int(0b0111111111111)):
                new_sig[i] = int(0b1110000) + int((sig >> 7) & 0b1111)
        elif sig < 0:
            sig = -sig
            if   (sig >= int(0b0000000000001)) & (sig <= int(0b0000000011111)):
                new_sig[i] = int(0b0000000) + int((sig >> 1) & 0b1111)         
            elif (sig >= int(0b0000000010000)) & (sig <= int(0b0000000111111)):
                new_sig[i] = int(0b0010000) + int((sig >> 1) & 0b1111)         
            elif (sig >= int(0b0000000100000)) & (sig <= int(0b0000001111111)):
                new_sig[i] = int(0b0100000) + int((sig >> 2) & 0b1111)         
            elif (sig >= int(0b0000001000000)) & (sig <= int(0b0000011111111)):
                new_sig[i] = int(0b0110000) + int((sig >> 3) & 0b1111)         
            elif (sig >= int(0b0000010000000)) & (sig <= int(0b0000111111111)):
                new_sig[i] = int(0b1000000) + int((sig >> 4) & 0b1111)         
            elif (sig >= int(0b0000100000000)) & (sig <= int(0b0001111111111)):
                new_sig[i] = int(0b1010000) + int((sig >> 5) & 0b1111)         
            elif (sig >= int(0b0001000000000)) & (sig <= int(0b0011111111111)):
                new_sig[i] = int(0b1100000) + int((sig >> 6) & 0b1111)         
            elif (sig >= int(0b0010000000000)) & (sig <=int(0b0111111111111)): 
                new_sig[i] = int(0b1110000) + int((sig >> 7) & 0b1111)         
            new_sig[i] = -new_sig[i]
        i += 1
    return new_sig

def descompress(signal):
    '''
    desfaz a compressão da lei A
    :param signal: sinal comprimido
    :return: sinal
    '''
    signal = signal.astype(np.int16)
    new_sig = np.zeros(len(signal)).astype(np.int16)
    i = 0
    for sig in signal:
        if sig >= 0:
            if (sig >= int(0b0000000)) & (sig <= int(0b0001111)):
                new_sig[i] = int(0b000000000000) + int((sig & 0b1111) << 1)+ int(0b1 << 0)
            elif (sig >= int(0b0010000)) & (sig <= int(0b0011111)):
                new_sig[i] = int(0b000000100000) + int((sig & 0b1111) << 1)+ int(0b1 << 0)
            elif (sig >= int(0b0100000)) & (sig <= int(0b0101111)):
                new_sig[i] = int(0b000001000000) + int((sig & 0b1111) << 2)+ int(0b1 << 1)
            elif (sig >= int(0b0110000)) & (sig <= int(0b0111111)):
                new_sig[i] = int(0b000010000000) + int((sig & 0b1111) << 3)+ int(0b1 << 2)
            elif (sig >= int(0b1000000)) & (sig <= int(0b1001111)):
                new_sig[i] = int(0b000100000000) + int((sig & 0b1111) << 4)+ int(0b1 << 3)
            elif (sig >= int(0b1010000)) & (sig <= int(0b1011111)):
                new_sig[i] = int(0b001000000000) + int((sig & 0b1111) << 5)+ int(0b1 << 4)
            elif (sig >= int(0b1100000)) & (sig <= int(0b1101111)):
                new_sig[i] = int(0b010000000000) + int((sig & 0b1111) << 6)+ int(0b1 << 5)
            elif (sig >= int(0b1110000)) & (sig <= int(0b1111111)):
                new_sig[i] = int(0b100000000000) + int((sig & 0b1111) << 7)+ int(0b1 << 6)
        elif sig < 0:
            sig = -sig
            if (sig >= int(0b0000000)) & (sig <= int(0b0001111)):                          
                new_sig[i] = int(0b000000000000) + int((sig & 0b1111) << 1)+ int(0b1 << 0) 
            elif (sig >= int(0b0010000)) & (sig <= int(0b0011111)):                        
                new_sig[i] = int(0b000000100000) + int((sig & 0b1111) << 1)+ int(0b1 << 0) 
            elif (sig >= int(0b0100000)) & (sig <= int(0b0101111)):                        
                new_sig[i] = int(0b000001000000) + int((sig & 0b1111) << 2)+ int(0b1 << 1) 
            elif (sig >= int(0b0110000)) & (sig <= int(0b0111111)):                        
                new_sig[i] = int(0b000010000000) + int((sig & 0b1111) << 3)+ int(0b1 << 2) 
            elif (sig >= int(0b1000000)) & (sig <= int(0b1001111)):                        
                new_sig[i] = int(0b000100000000) + int((sig & 0b1111) << 4)+ int(0b1 << 3) 
            elif (sig >= int(0b1010000)) & (sig <= int(0b1011111)):                        
                new_sig[i] = int(0b001000000000) + int((sig & 0b1111) << 5)+ int(0b1 << 4) 
            elif (sig >= int(0b1100000)) & (sig <= int(0b1101111)):                        
                new_sig[i] = int(0b010000000000) + int((sig & 0b1111) << 6)+ int(0b1 << 5) 
            elif (sig >= int(0b1110000)) & (sig <= int(0b1111111)):                        
                new_sig[i] = int(0b100000000000) + int((sig & 0b1111) << 7)+ int(0b1 << 6) 
            new_sig[i] = -new_sig[i]
            
        i += 1
    return new_sig

def companding(signal):
    '''
    Amplica a lei A em um sinal, realizando um companding
    :param signal: sinal
    :return: sinal após todas as etapas para companding
    '''
    sinal_quant = conversao_ad(signal)
    sin_wave_alaw = encode_a_law(sinal_quant)
    sinal_compress = compress(sin_wave_alaw)
    sinal_descompress = descompress(sinal_compress)
    sin_wave_companding = decode_a_law(sinal_descompress)
    return sin_wave_companding

#constantes para criar gráfico da senoide
seconds = 1
fs = 500

time = np.linspace(0, seconds, seconds*fs)
sin_wave = amp*np.sin(2 * np.pi * fs * time)
sin_wave = sin_wave.astype(np.int16)

sinal_quant = conversao_ad(sin_wave)
sin_wave_alaw = encode_a_law(sinal_quant)
sinal_compress = compress(sin_wave_alaw)
sinal_descompress = descompress(sinal_compress)
sin_wave_companding = decode_a_law(sinal_descompress)

# ploting
plt.figure(figsize=(10,10))
plt.plot(time, sin_wave/amp, label = 'wave')
plt.plot(time, sin_wave_alaw/amp, label = 'alaw')
#plt.plot(time, sinal_compress, label = 'compress')
plt.plot(time, sinal_descompress/amp, label = 'compress + descompress')
plt.plot(time, sin_wave_companding/amp, label = 'companding')
plt.legend()
plt.xlabel('tempo,segundos')
plt.savefig('sin', bbox_inches='tight')

# ------------------------------------ Audio

fsrec = 44100  # Audio sample rate
rec_seconds = 1 # recording time
time_recording = np.linspace(0, rec_seconds, rec_seconds*fsrec) # x axis for ploting audio
myrecording = sd.rec(int(rec_seconds * fsrec), samplerate=fsrec, channels=2, dtype='int16')
sd.wait()  # Wait until recording is finished

myrecording_lin = myrecording >> 3 # Bit shift
myrecording_companding = np.full_like(myrecording_lin, 0) #create a equal matrix, same dimensions with 0 values

myrecording_companding[:,0] = companding(myrecording_lin[:,0])
myrecording_companding[:,1] = companding(myrecording_lin[:,1])

myrecording_companding = myrecording_companding << 3 # Bit shift

#save files
write('myrecording.wav', fsrec, myrecording)  # Save as WAV file
write('myrecording_companding.wav', fsrec, myrecording_companding)  # Save as WAV file


# ploting
fig, (ax1, ax2) = plt.subplots(2, figsize=(16,12))
fig.suptitle('Audio plots')
ax1.plot(time_recording, myrecording[:,0], 'k', label = 'audio original')
ax2.plot(time_recording, myrecording_companding[:,0], 'r', label = 'audio após companding')
fig.legend()
plt.xlabel('tempo,segundos')
plt.savefig('sound', bbox_inches='tight') #check your folder