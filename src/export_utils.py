import numpy as np

matriu = np.load('bassiero.npy')

# Guarda la matriu en un fitxer de text llegible
# fmt='%.3f' assegura que es guardin amb 3 decimals en lloc de notació científica
np.savetxt('matriu_exportada.csv', matriu, delimiter=',', fmt='%.3f')

print("Matriu exportada correctament a CSV!")