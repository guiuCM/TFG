# Anàlisi i Debugging de pendent6.py
**Context:** Comparació entre algoritmes de generació TIN basat en error d'alçada (original.py) vs error angular (pendent6.py)

---

## Resum Executiu

L'algoritme `pendent6.py` (basat en error angular entre normals) produeix **pitjors resultats** que `original.py` (basat en error d'alçada absoluta) en termes de precisió d'alçades:

- **500 punts:** RMSE 96.76m (pendent6) vs 13.66m (original) - **7x pitjor**
- **2000 punts:** RMSE 41.70m (pendent6) vs 6.60m (original) - **6.3x pitjor**

---

## Problemes Identificats i Resolts

### 1. **Normalització incorrecta dels normals del grid** ✅ RESOLT

**Problema:**
```python
nz = 1  # Escalar, no array!
norm = np.sqrt(nx**2 + ny**2 + nz**2)
```

Això creava normals amb magnituds entre 1.0 i 25.85, no normalitzats.

**Solució:**
```python
nz = np.ones_like(nx)  # Array amb la mateixa forma
norm = np.sqrt(nx**2 + ny**2 + nz**2)
```

**Resultat:** Normals correctament normalitzats (magnitud = 1.0 ± 1e-7)

---

### 2. **Clustering de punts** ✅ PARCIALMENT RESOLT

**Problema:**
L'algoritme greedy afegia punts consecutius en la mateixa zona (índexs 626, 627, 628 en la mateixa fila).

**Solució implementada:**
```python
min_distance = self.step * self.pixel_size * 2  # Distància mínima = 2 pixels (4m)

for idx in sorted_indices:
    candidate_xy, _ = self._get_coords_from_index(candidate_indices[idx])
    distances = np.linalg.norm(self.tin.points - candidate_xy, axis=1)
    
    if np.all(distances >= min_distance):
        # Acceptar aquest candidat
        break
```

**Resultat:** Distància mínima entre punts: 4.0m ✓

---

### 3. **Errors angulars >90°: Normal o anòmal?** ⚠️ COMPORTAMENT ESPERAT

**Observacions:**
- Errors reportats durant iteracions: 113-153°
- Després de debug: errors amb dot product negatiu

**Exemple real:**
```
Grid normal: [-0.964, -0.215, 0.159]  → Pendent molt pronunciat (quasi vertical)
TIN normal:  [ 0.801, -0.066, 0.595]  → Triangle més pla
Dot product: -0.565
Angle: 124.39°
```

**Causa identificada:**
Els errors >90° són **correctes i esperats** quan:
1. El terreny té pendents molt pronunciats (Z-component petit)
2. El TIN té triangles grans i plans (Z-component gran)
3. Les components X,Y apunten en direccions oposades

**Interpretació:**
- Angle >90° significa que el TIN està capturant **molt malament** la geometria local
- Això hauria de forçar l'algoritme a afegir més punts en aquestes zones
- **PERÒ** encara així els resultats finals són dolents

---

## Per què pendent6.py falla?

### Hipòtesi 1: Error angular ≠ Error d'alçada

**Teoria:**
Optimitzar per error angular (diferència entre normals) **no és equivalent** a optimitzar per error d'alçada (diferència entre elevacions).

**Exemple conceptual:**
- Una muntanya amb pendent constant de 45° però mal situada en alçada absoluta tindrà:
  - **Error angular baix** (les normals coincideixen)
  - **Error d'alçada alt** (està desplaçada verticalment)

**Evidència:**
```
Pendent6 (angular):
- RMSE Pendent: 0.786 (acceptable)
- RMSE Alçada: 96.76m (MOLT dolent)

Original (alçada):
- RMSE Pendent: 0.612 (millor!)
- RMSE Alçada: 13.66m (MOLT millor)
```

**Conclusió:** Minimitzar error angular **no garanteix** minimitzar error d'alçada.

---

### Hipòtesi 2: Els triangles de Delaunay són massa grans

**Observació:**
Amb només 500 punts en un grid de 1500×1500 (2.25 milions de punts originals):
- Cada punt del TIN representa ~4500 punts del grid
- Els triangles són enormes (àrea mitjana ~18,000 m²)
- Un triangle pla no pot capturar la curvatura local del terreny real

**Efecte:**
Fins i tot amb errors angulars >100° correctament identificats, afegir **un sol punt** per iteració és insuficient per refinar zones crítiques.

---

### Hipòtesi 3: La inicialització és inadequada

**Inicialització actual:**
1. 4 cantonades del grid
2. 1 punt de màxim residual (basat en pla ajustat a cantonades)

**Problema:**
El punt 5 es selecciona basant-se en error d'**alçada** (residual d'un pla), no en error angular. Això pot crear un biaix inicial que l'algoritme no pot corregir.

---

## Experiments de validació pendents

### Test 1: Comparació directa amb mateix nombre de punts
```python
# Amb 100 punts:
pendent6: RMSE = ? m
original: RMSE = ? m
```

### Test 2: Verificar si la selecció de punts és coherent
```python
# Visualitzar distribució espacial dels punts seleccionats
# Hipòtesi: pendent6 concentra punts en zones de transició (ridges/valleys)
#           mentre original distribueix més uniformement
```

### Test 3: Error angular vs error d'alçada per cada punt
```python
# Correlació entre:
# - angular_error[i] 
# - height_error[i]
# Si correlació és baixa → explica el problema
```

---

## Conclusions provisionals

1. **Tècnicament correcte:** El codi de `pendent6.py` calcula correctament els errors angulars després de la correcció de normalització.

2. **Conceptualment qüestionable:** Optimitzar per error angular **no** implica optimitzar per error d'alçada, que és la mètrica que volem minimitzar en moltes aplicacions.

3. **La hipòtesi original està parcialment equivocada:**
   - ✅ Cert: "Una muntanya es defineix per la seva forma" (per anàlisi geomòrfica)
   - ❌ Fals: "Per tant, minimitzar error de forma minimitzarà error d'alçada"

4. **Possible solució:** Enfocament híbrid que combini ambdós criteris:
   ```python
   combined_error = alpha * angular_error + beta * height_error
   ```

---

## ACTUALITZACIÓ: Objectiu real del projecte

**L'usuari NO vol minimitzar error d'alçada**, sinó **maximitzar fidelitat geomètrica amb pocs punts**.

### Resultats amb mètrica correcta (error angular)

```
Amb 200 punts:
- pendent6: Error angular mitjà 31.5° 
- original: Error angular mitjà 22.9°
```

**Conclusió sorprenent:** Fins i tot optimitzant PER error angular, `original.py` dona millors resultats!

### Causa identificada: Estratègia greedy ineficient

**Problema:**
L'algoritme greedy afegeix **un sol punt per iteració**, però:
1. Afegir 1 punt en una zona gran crea triangles que segueixen sent massa grans
2. Els punts veïns encara tenen errors alts
3. L'algoritme els selecciona consecutivament (clustering local)

**Evidència:**
```
Iteracions 1-4 afegeixen punts a:
  (626,1252), (627,1252), (628,1252) - MATEIXA COLUMNA!
```

### Solució proposada: Batch selection

En lloc d'afegir 1 punt per iteració, afegir N punts simultàniament:
- Seleccionar top-K punts amb més error
- Filtrar per distància mínima entre ells
- Afegir tots alhora al TIN
- Recalcular errors

Això evitaria el clustering i distribuiria millor els punts.

## Qüestions obertes

1. ~~**És realment l'error angular inadequat**, o hi ha un bug que encara no hem trobat?~~  
   **RESOLT:** L'algoritme funciona correctament. El problema és l'estratègia greedy.

2. **Quin batch size és òptim?** (10 punts? 50 punts?)

3. **Com equilibrar exploració (punts distribuïts) vs explotació (refinar zones crítiques)?**

4. **Necessitem millor inicialització?** (grid 3×3 en lloc de 4 corners + 1)

---

## Referències de codi

### Càlcul de normals del grid (pendent6.py:36-46)
```python
dz_dy, dz_dx = np.gradient(self.h_grid, spacing, spacing)
nx = -dz_dx
ny = -dz_dy
nz = np.ones_like(nx)  # ← FIX aplicat

norm = np.sqrt(nx**2 + ny**2 + nz**2)
self.normal_grid = np.dstack((nx/norm, ny/norm, nz/norm))
```

### Càlcul d'error angular (pendent6.py:51-98)
```python
# Normals del TIN
u = p1 - p0
v = p2 - p0
tin_normals = np.cross(u, v)
tin_normals = tin_normals / np.linalg.norm(tin_normals, axis=1, keepdims=True)

# Assegurar orientació cap amunt
flip_vec = tin_normals[:, 2] < 0
tin_normals[flip_vec] *= -1

# Error angular
dot_product = np.einsum('ij,ij->i', real_normals, tin_normals)
dot_product = np.clip(dot_product, -1.0, 1.0)
angles_deg = np.degrees(np.arccos(dot_product))
```

### Filtrat espacial (pendent6.py:346-358)
```python
min_distance = self.step * self.pixel_size * 2

for idx in sorted_indices:
    worst_local_idx = idx
    candidate_flat_idx = cand_arr[worst_local_idx]
    candidate_xy, candidate_z = self._get_coords_from_index(candidate_flat_idx)
    
    distances = np.linalg.norm(self.tin.points - candidate_xy, axis=1)
    
    if np.all(distances >= min_distance):
        break
```

---

## Estat actual

**Bugs resolts:**
- ✅ Normalització de normals del grid
- ✅ Clustering espacial de punts

**Problema persistent:**
- ❌ RMSE d'alçada 7x pitjor que original.py

**Següent pas recomanat:**
Crear test de correlació entre error angular i error d'alçada per validar/refutar la hipòtesi principal.
