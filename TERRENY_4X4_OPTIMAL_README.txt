
TERRENY 4x4 ÒPTIM PARA AMBDÓS ALGORISMES
==========================================

Fitxer: terrain_4x4_optimal.npy

DESCRIPCIÓ:
-----------
Terreny 4x4 amb estructura de turons gaussians suaus. Creat específicament
per produir resultats equilibrats en:
  1. original.py (error alçada)
  2. pendent7.py (error angular)

CARACTERÍSTICAS:
----------------
- Dimensions: 4x4 punts
- Alçades: [2.046, 6.103] m
- Estructura: Turons suaus + vall local
- Realisme: Alt (similar a topografia natural)

RESULTATS ESPERATS:
------------------
original.py (error alçada):
  - Error baix (0.1-0.3 m per 6-8 punts)
  - Convergència ràpida

pendent7.py (error angular):
  - Error angular baix (1-3° per 12-14 punts)
  - Comportament gradual

ÚS:
---
python3 comparacio_bona.py
# Canvia FILENAME = 'terrain_4x4_optimal.npy'
