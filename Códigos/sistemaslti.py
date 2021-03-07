import pandas as pd
import matplotlib.pyplot as plt

obitos = pd.read_csv("data.csv", sep=';')['obitosNovos']

h = []
h.append(0)
h.append(0)
h.append(0)
h.append(0)
h.append(0)
h.append(0)
h.append(0)
h.append(0)

for n in range(8,272):
    h.append((1/8)*(obitos[n] + obitos[n - 1] + obitos[n - 2] + obitos[n - 3] + obitos[n - 4] + obitos[n - 5] + obitos[n - 6] + obitos[n - 7]))

plt.figure(figsize=(10, 10))
plt.plot(obitos[0:272], color = 'green', label='Dados de Óbitos')
plt.plot(h, color = 'red', label='Dados Filtratos')
plt.title('Óbitos')
plt.legend()

plt.savefig('Figure_1.png')
plt.show()