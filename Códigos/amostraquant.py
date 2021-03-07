import numpy as np
# amostraquant.py
# Downsample the signal, selecting every nth element.
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
    factor = int(ts/td)
    s_out=downsample(sig_in,factor)
    s_out=upsample(s_out,factor)

    return s_out

def quantizacao(sinal,amp,n):
    # quantizacao
    Delta=2*amp/n  # quantizacao uniforme(Delta)
    x = sinal + np.array(amp)  # somar nivel CC igual a amp
    q = np.round(x/Delta)  # dividir em intervalos iguais a D
    x = Delta/2 + Delta*q - np.array(amp)  # quantizar e remover nivel CC
    return x

def amostra_quant(sig_in,td,ts, amp, L):

    #if(np.remainder(ts/td, 1)==0):
    nfac=int(ts/td)
    s_out=downsample(sig_in,nfac)
    sq_out=quantizacao(s_out, amp, L)
    s_out=upsample(s_out,nfac)
    sq_out=upsample(sq_out,nfac)
    #else:
    #    warning('Erro! ts/td não é um inteiro!')
    #    s_out=[]
    #    sq_out=[]

    return s_out,sq_out

def uniquan(sig_in,L):
    # L - numero de niveis de quantizacao uniforme
    # sig_in - vetor para sinal de entrada
    sig_pmax = max(sig_in)  # pico positivo
    sig_nmax = min(sig_in)  # pico negativo

    Delta=(sig_pmax - sig_nmax)/L  # intervalo de quantizacao
    levelvector = np.linspace(Delta/2, Delta, Delta * sig_pmax)
    q_level = sig_nmax+levelvector - Delta/2  # define Q niveis


    L_sig = len(sig_in)  # comprimento do sinal
    sigp = (sig_in-sig_nmax)/Delta+1/2  # converte a faixa de 1/2 a L+1/2
    qindex=round(sigp)  # arredonda a 1,2,...,L niveis
    qindex=min(qindex,L)  # elimina L+1, se houver
    q_out=q_level(index)  # usa vetor index para gerar saida
    SQNR=20*log10(norm(sig_in)/norm(sig_in-q_out))  # valor da SQNR
    
    return q_out, Delta, SQNR

def bi2de(binary):
    # Same basic function as matlab bi2de function
    # Convert a binary array to a decimal number
    # Read from right to left in binary array
    bin_temp = 0
    bin_res = np.zeros(len(binary), dtype=int)
    for i in range(len(binary)):
        for j in range(len(binary[i])):
            bin_temp = bin_temp + binary[i][j] * (2 ** j)
        bin_res[i] = bin_temp
        bin_temp = 0
    return bin_res