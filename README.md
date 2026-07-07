# GRID-TIN Navigation Mesh Generator

Aquest projecte és part d'un Treball de Final de Grau (TFG) de la Facultat d'Informàtica de Barcelona (FIB). L'objectiu principal és desenvolupar un algorisme eficient per a la generació de malles de navegació a partir de mapes d'alçades (heightfields) en entorns exteriors complexos.

## Descripció
El sistema implementa una estratègia incremental de GRID a TIN (Triangulated Irregular Network) que transforma mapes d'alçades en xarxes de triangles optimitzades. A diferència dels mètodes tradicionals, aquest enfocament permet una representació més precisa del terreny per a la navegació d'agents intel·ligents en videojocs o simulacions, optimitzant la densitat de triangles basant-se en l'error del pendent.

## Característiques principals
- **Conversió GRID-TIN:** Transformació de matrius d'alçades (heightfields) a malles poligonals optimitzades.
- **Estratègia Incremental:** Us d'un algorisme voraç (greedy).
- **Optimització Geomètrica:** Ús de la triangulació de Delaunay per garantir triangles ben condicionats i evitar la formació de triangles "estella" o "slivers".
- **Anàlisi Semàntica:** Comparativa de mètriques d'error (RMSE, MAE i Error Màxim) entre l'alçada i el pendent amb els diferents algorismes.
- **Probes visuals:** Videos que mostren els errors visualment dels algorismes tant d'alçada com amb l'error angular.

## Tecnologies utilitzades
- **Llenguatge:** Python 3
- **Llibreries:** NumPy, SciPy (per a triangulació i càlculs numèrics), Matplotlib (per a visualització).

## Algorismes
- **original.py:** ALgorisme de GRID a TIN basat amb l'error d'alçada.
- **pendent7.py:** ALgorisme de GRID a TIN basat amb l'error angular, canviant els diferents modes o el resultat que retorna la funció de ponderació es poden arribar a tots els algorismes que comento a la memòria.

## Tests
- **bassiero.npy:** Matriu d'alçades dels pic de bassiero als pirineus (1500x1500).
- **prova_oscar.npy:** Matriu 4x4 per probar el funcionament de l'algorisme.
- **terrain16.npy:** Matriu 4x4 per probar el funcionament de l'algorisme.
