import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import square, sawtooth

path = '../imagens/'

Ts = 16  # pontos por simbolo
Tb = int(Ts/2)  # tempo de duracao do bit
Nb = 10000  # numero de bits
B = np.random.randint(2, size=Nb) # gera Nb bits aleatoriamente
T = np.linspace(1, 50 * Tb, 50 * Tb)  # tempo usado para mostrar grafico

# pulsos
# pulso NRZ
pulso_nrz = np.ones(Ts)

# pulso RZ
pulso_rz = np.zeros(Ts)
pulso_rz[int(Ts/2): Ts] = np.ones(int(Ts/2))

# pulso = pulso_nrz
pulso = pulso_nrz
pulsos = 'pulso_nrz'

# Codigo unipolar NRZ
NRZuni = []
for b in B:
    if (b == 0):
        simbolo = 0
    else:
        simbolo = 1
    NRZuni = np.concatenate([NRZuni, simbolo * pulso])

plt.figure(figsize=(16, 9), )
#plt.stem(,,'r',)
plt.plot(T, NRZuni[0:len(T)],'k')
#plt.xlabel('')
plt.axis([0,len(T), -2, 2])
plt.title('NRZ')
plt.savefig(path + pulsos + '1bNRZuni', bbox_inches='tight')
plt.close()

# Codigo polar NRZ
NRZ = []
for b in B:
    if (b == 0):
        simbolo = -1
    else:
        simbolo = 1
    NRZ = np.concatenate([NRZ, simbolo * pulso])

plt.figure(figsize=(16, 9), )
#plt.stem(,,'r',)
plt.plot(T, NRZ[0:len(T)],'k')
#plt.xlabel('')
plt.axis([0,len(T), -2, 2])
plt.title('NRZ')
plt.savefig(path + pulsos + '1NRZ', bbox_inches='tight')
plt.close()

# codigo NRZI
NRZI = []

simbolo = -1
for b in B:
    if (b == 1):
        simbolo = simbolo * (-1)

    NRZI = np.concatenate([NRZI, simbolo * pulso])

plt.figure(figsize=(16, 9), )
#plt.stem(,,'r',)
plt.plot(T, NRZI[0:len(T)],'k')
#plt.xlabel('')
plt.axis([0,len(T), -2, 2])
plt.title('NRZI')
plt.savefig(path + pulsos + '2NRZI', bbox_inches='tight')
plt.close()

## Bipolar AMI
AMI = []

aux = 1
for b in B:
    if (b == 1):
        if (aux == 1):
            simbolo = 1
            aux = 0
        else:
            simbolo = -1
            aux = 1
    else:
        simbolo = 0

    AMI = np.concatenate([AMI, simbolo * pulso])

# grafico
plt.figure(figsize=(16, 9), )
#plt.stem(,,'r',)
plt.plot(T, AMI[0:len(T)],'k')
#plt.xlabel('')
plt.axis([0,len(T), -2, 2])
plt.title('AMI')
plt.savefig(path + pulsos + '3AMI', bbox_inches='tight')
plt.close()


# PSEUDOTERNARIA
PSEUD = []

aux = 1
i = 0
for b in B:
    if (b == 0):
        if (aux == 1):
            simbolo = 1
            aux = 0
        else:
            simbolo = -1
            aux = 1
    else:
        simbolo = 0

    PSEUD = np.concatenate([PSEUD, simbolo * pulso])

plt.figure(figsize=(16, 9), )
#plt.stem(,,'r',)
plt.plot(T, PSEUD[0:len(T)],'k')
#plt.xlabel('')
plt.axis([0,len(T), -2, 2])
plt.title('Pseudoternaria')
plt.savefig(path + pulsos + '4Pseudoternaria', bbox_inches='tight')
plt.close()

#MANCHESTER
MANCH = []

m = 1
for b in B:
    if (b == 0):
        simbolo = 1
    else:
        simbolo = -1

    MANCH = np.concatenate([MANCH, simbolo * pulso[0: int(Ts / 2)] , (-1) * simbolo * pulso[int(Ts / 2 + 1): Ts]])

plt.figure(figsize=(16, 9), )
# plt.stem(,,'r',)
plt.plot(T, MANCH[0:len(T)], 'k')
# plt.xlabel('')
plt.axis([0, len(T), -2, 2])
plt.title('MANCH')
plt.savefig(path + pulsos + '5MANCH', bbox_inches='tight')
plt.close()


#MANCHESTER DIFERENCIAL

DIFFMANCH = []
aux = -1
for b in B:
    if (b == 0):
        simbolo = aux
    elif (b == 1) & (aux == -1):
        simbolo = 1
        aux = 1
    else:
        simbolo = -1
        aux = -1

    DIFFMANCH = np.concatenate([DIFFMANCH, simbolo * pulso[0:int(Ts / 2)] , (-1) * simbolo * pulso[int(Ts / 2 + 1): Ts]])

plt.figure(figsize=(16, 9), )
# plt.stem(,,'r',)
plt.plot(T, DIFFMANCH[0:len(T)], 'k')
# plt.xlabel('')
plt.axis([0, len(T), -2, 2])
plt.title('DIFFMANCH')
plt.savefig(path + pulsos + '6DIFFMANCH', bbox_inches='tight')
plt.close()

#B8ZS
B8ZS = []
aux = -1
zero = 0
rule = [-1,1,0,1,-1]
i = 0
rule_i = 0
trigger_rule = False
signal_rule = 1
for b in B:
    if (b == 1):
        zero = 0
        if (aux == 1):
            simbolo = 1
            aux = 0
        else:
            simbolo = -1
            aux = 1
    else:
        zero += 1
        simbolo = 0

    try:
        if (zero == 4) & (B[i+1] == 0) & (B[i+2] == 0) & (B[i+3] == 0) & (B[i+4] == 0):
            trigger_rule = True
            rule_i = 0

        if (B[i+1] == 0):
            signal_rule = 1
        else:
            signal_rule = -1
    except:
        pass

    i += 1

    if trigger_rule:
        simbolo = rule[rule_i] * signal_rule
        rule_i += 1
        zero = 0

    if rule_i == len(rule):
        trigger_rule = False
        rule_i = 0
        zero = 0

    B8ZS = np.concatenate([B8ZS, simbolo * pulso])

plt.figure(figsize=(16, 9), )
# plt.stem(,,'r',)
plt.plot(T, B8ZS[0:len(T)], 'k')
# plt.xlabel('')
plt.axis([0, len(T), -2, 2])
plt.title('B8ZS')
plt.savefig(path + pulsos + '7B8ZS', bbox_inches='tight')
plt.close()

#HDB3

HDB3 = []
aux = -1

i = 0
rule_i = 0
trigger_rule = False
secondary_trigger = False
rule_simbol = 0
signal_rule = 1
simbolos = []
counter = 0
for b in B:
    if (b == 1):
        zero = 0
        if (aux == 1):
            simbolo = 1
            aux = 0
        else:
            simbolo = -1
            aux = 1
    else:
        simbolo = 0

    simbolos.append(simbolo)

    try:
        if (B[i-1] == 0) & (B[i-2] == 0) & (B[i-3] == 0) & (B[i] == 0):
            trigger_rule = True
            rule_simbol = simbolos[i-4]
        if  secondary_trigger & (B[i+1] == 0) & (B[i+2] == 0) & (B[i+3] == 0) & (B[i+4] == 0):
            trigger_rule = True
            for c in range(0, len(simbolos)):
                if(simbolos[i-c] != 0):
                    rule_simbol = simbolos[i-c]

        if (counter % 2 == 0):
            par = True
        else:
            par = False
    except:
        pass

    i += 1
    counter += 1

    if trigger_rule:
        if secondary_trigger & (rule_i == 0 | rule_i == 3):
            simbolo = rule_simbol
            rule_i += 1
        else:
            simbolo = rule_simbol
            trigger_rule = False

        secondary_trigger = True
        counter = 0

    if rule_i == 4:
        trigger_rule = False
        rule_i = 0

    HDB3 = np.concatenate([HDB3, simbolo * pulso])


plt.figure(figsize=(16, 9), )
# plt.stem(,,'r',)
plt.plot(T, HDB3[0:len(T)], 'k')
# plt.xlabel('')
plt.axis([0, len(T), -2, 2])
plt.title('HDB3')
plt.savefig(path + pulsos + '8HDB3', bbox_inches='tight')
plt.close()

# Calculo de PSDs
# PSD usando metodo de Welch (pode ser usado tambem o periodogram)

from scipy.signal import welch, windows

plt.figure(figsize=(16, 9), )

# Hpsd=psd(spectrum.welch,NRZ) # matlab
win = windows.hann(1024, True)
[f, Hpsd] = welch(NRZ, window=win, detrend = False)
plt.plot(Hpsd, 'r', label = 'NRZ Polar')

win = windows.hann(1024, True)
# Hpsd=psd(spectrum.welch,MANCH) # matlab
[f, Hpsd] = welch(MANCH, window=win, detrend = False)
plt.plot(Hpsd, 'b', label = 'MANCH')

win = windows.hann(1024, True)
[f, Hpsd] = welch(AMI, window=win, detrend = False)
plt.plot(Hpsd, 'g', label = 'AMI')

win = windows.hann(1024, True)
[f, Hpsd] = welch(DIFFMANCH, window=win, detrend = False)
plt.plot(Hpsd, 'y', label = 'DIFFMANCH')

plt.legend()
plt.xlabel('f')
plt.title('PSD')
plt.savefig(path + pulsos + '9PSD', bbox_inches='tight')
plt.close()

plt.figure(figsize=(16, 9), )

win = windows.hann(1024, True)
[f, Hpsd] = welch(AMI, window=win, detrend = False)
plt.plot(Hpsd, 'g', label = 'AMI')

win = windows.hann(1024, True)
[f, Hpsd] = welch(PSEUD, window=win, detrend = False)
plt.plot(Hpsd, 'm', label = 'PSEUD')

win = windows.hann(1024, True)
[f, Hpsd] = welch(B8ZS, window=win, detrend = False)
plt.plot(Hpsd, 'c', label = 'B8ZS')

win = windows.hann(1024, True)
[f, Hpsd] = welch(HDB3, window=win, detrend = False)
plt.plot(Hpsd, 'k', label = 'HDB3')

plt.legend()
plt.xlabel('f')
plt.title('PSD')
plt.savefig(path + pulsos + '9bPSD', bbox_inches='tight')
plt.close()

plt.figure(figsize=(16, 9), )

win = windows.hann(2048, True)
[f, Hpsd] = welch(AMI, window=win, detrend = False)
plt.plot(Hpsd, 'g', label = 'AMI')

win = windows.hann(2048, True)
[f, Hpsd] = welch(PSEUD, window=win, detrend = False)
plt.plot(Hpsd, 'm', label = 'PSEUD')

win = windows.hann(2048, True)
[f, Hpsd] = welch(B8ZS, window=win, detrend = False)
plt.plot(Hpsd, 'c', label = 'B8ZS')

win = windows.hann(2048, True)
[f, Hpsd] = welch(HDB3, window=win, detrend = False)
plt.plot(Hpsd, 'k', label = 'HDB3')

plt.legend()
plt.xlabel('f')
plt.title('PSD')
plt.axis((20,70,12,20))
plt.savefig(path + pulsos + '9bPSDzoom', bbox_inches='tight')
plt.close()

#NRZ unipolar x polar
plt.figure(figsize=(16, 9), )

# Hpsd=psd(spectrum.welch,NRZ) # matlab
win = windows.hann(1024, True)
[f, Hpsd] = welch(NRZuni, window=win, detrend = False)
plt.plot(Hpsd, 'r', label = 'NRZ Unipolar')

# Hpsd=psd(spectrum.welch,NRZ) # matlab
win = windows.hann(1024, True)
[f, Hpsd] = welch(NRZ, window=win, detrend = False)
plt.plot(Hpsd, 'b', label = 'NRZ Polar')
plt.axis((0,500,0,40))
plt.legend()
plt.xlabel('f')
plt.title('PSD')
plt.savefig(path + pulsos + '9cPSDnrz', bbox_inches='tight')
plt.close()