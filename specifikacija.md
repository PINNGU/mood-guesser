# Specifikacija Projekta - MoodGuesser-MoodSync

**Ime i prezime:** Nikola Cvijetinovic
**Broj indeksa:** RA-134/2021

---

## 2. Zadatak

### Šta radimo?

Pravimo sistem gde 5 muzičkih platformi (tipa Spotify, Apple Music) zajedno treniraju AI model koji prepoznaje raspoloženje u pesmama - ali **ne dele** podatke o korisnicima između sebe.

**Zašto?** Privatnost korisnika + poslovna tajna.

---

## 3. Federativno učenje

### Algoritam

**FedAvg** - svaka platforma trenira kod sebe, šalje samo brojeve (parametre modela), ne šalje pesme.

### Dataset

**Spotify Tracks** sa Kaggle-a - ~114,000 pesama sa karakteristikama (tempo, energy, valence...).

### Raspoloženja (5 klasa)

1. **happy** - veselo
2. **sad** - tužno  
3. **energetic** - energično
4. **relaxed** - opušteno
5. **neutral** - neutralno

Pravimo ih na osnovu Spotify karakteristika (npr. ako je `valence > 0.6` i `energy > 0.5` = happy).

### Kako delimo podatke

5 platformi, svaka dobija ~20% pesama, ali različite žanrove:
- Platforma 0: više happy/energetic
- Platforma 1: više sad/relaxed
- Platforma 2: skroz energetic
- Platforma 3: balansirana
- Platforma 4: više relaxed/neutral

### Šta merimo

- **Accuracy** - procenat tačnih predviđanja
- **Grafik** - kako tačnost raste kroz 20 rundi
- **Poređenje** - federativno vs centralizovano treniranje

---

## 4. CRDT

### Zašto?

3 koordinatora rade istovremeno. Moraju imati isto stanje bez konflikata.

### Koje koristimo

**G-Counter** - broji pesme po raspoloženju  
**OR-Set** - lista svih raspoloženja  
**LWW-Map** - čuva verziju modela (pobeđuje novija)

Merge se dešava svakih 5 sekundi između koordinatora.

---

## 5. Clustering

### Šta imamo

3 koordinatora (port 8001, 8002, 8003) u Docker kontejnerima.

### Šta rade

- Primaju update-e od platformi
- Rade prosek (FedAvg)
- Vraćaju novi model

### Fault tolerance

Ako padne koordinator:
```
docker kill coordinator-2
```
→ Platforme se prebace na drugi koordinator  
→ Sistem nastavlja normalno

---

## 6. Tehnologije

**Jezik:** Python 3.10

**Biblioteke:**
- PyTorch - neuronska mreža
- FastAPI - REST API za koordinatore
- Docker - kontejneri
- Pandas/NumPy - obrada podataka

**Arhitektura modela:**  
7 input → 64 hidden → 64 hidden → 5 output

---

## 7. Pokretanje
```bash
# 1. Skini dataset i stavi u data/spotify_tracks.csv

# 2. Pokreni koordinatore
docker-compose up -d

# 3. Pokreni eksperiment
python run_experiment.py --rounds 20

# 4. Ugasi koordinator (demo)
docker kill coordinator-2
# Sistem nastavlja da radi!
