# INTERACTIVNA KONZOLA, SKRIPTA, RUN

#%% PRINT
print('hello','world')               # ispisivanje na konzolu
moje_ime = 'Milos'                   # dodeljivanje vrednosti promenljivoj
print('hello', moje_ime)             # ispisivanje vise vrednosti

#%% ARITMETIKA I MATH BIBLIOTEKA
a = 10/4
print(a**2)                          # kvadriranje
print(a, a//2, a%2)                  # celobrojno deljenje i ostatak
import math                          # uvozenje biblioteke za matematicke operacije
print(math.log(a))                   # logaritmovanje (funkcija iz biblioteke)

#%% NIZOVI
niz = [1,2]                          # pravljenje niza brojeva
niz.append(3)                        # dodavanje elementa u niz
niz = ['a','b','c']                  # pravljenje niza stringova
niz.remove('b')                      # izbacivanje elementa na osnovu vrednosti
niz.pop(0)                           # izbacivanje elemente na osnovu pozicije
len(niz)                             # duzina niza
print(niz[0])                        # uzimanje prvog elementa niza
print(niz[1:3])                      # uzimanje podniza od pozicije 1 do pozicije 2
print(niz[-1])                       # uzimanje poslednjeg elementa niza

#%% DICTIONARY (Recnik)
godiste = {}                         # inicijalizacija recnika (kolekcije kljuc:vrednost)
godiste['Milos']=1982                # ubacivanje vrednosti 1982 pod kljucem 'Milos'
godiste['Dusan']=1992
print(godiste['Milos'])              # uzimanje vrednosti pod kljucem 'Milos'
ocene = {}
ocene['Milos'] = [5,4,6]             # dodeljivanje niza kao vrednosti za kljuc 'Milos' (vrednosti mogu biti sta)
ocene['Milos'] = {'RAMU':9, 'SPI':8} # dodeljivanje recnika kao vrednosti
print(ocene['Milos']['RAMU'])        # uzimanje vrednosti za kljuc 'Milos' iz recnika (ocene), pa iz rezultata uzeti vrednost pod kljucem 'RAMU'
print('Milos' in ocene)              # ispitivanje da li se kljuc 'Milos' nalazi u recniku

#%% FUNKCIJA I POZIVI
def stepen_broja(broj, stepen):      # definicija funkcije i ulaznih parametara
    kvadrat = broj**stepen           # telo funkcije: manipulacija parametrima
    return kvadrat                   # vracanje rezultata funkcije

print(stepen_broja(3,3))             # poziv funkcije sa parametrima

#%% IF, FOR LOOP & RANGE
a = 20
if a<10:                             # uslovno izvrsavanje
    print(a,'je manje od 10')
else:                                # ukoliko uslov nije zadovoljen
    print(a,'je vece ili jednako od 10')
if ocene['Milos']['RAMU']>5:
    print('polozio si')

for i in [1,2,5,6]:                  # petlja koja iterira kroz sve vrednosti kolekcije
    print(i)
for rec in ['kako','je','lepo','programirati']:
    if rec=='lepo':
        rec='divno'
    print(rec)
for i in range(2,20):                # petlja koja iterira od broja 2 do 19
    print(i)

#%% -------------------------------------- ZADATAK
# Napisati funkciju koja prima niz stringova (reci), a vraca recnik koji kao vrednosti ima duzine reci sa ulaza. I koji preskace prazne reci
def vrati_recnik(reci):
    recnik = {}
    for rec in reci:
        if rec == '':
            continue
        recnik[rec] = len(rec)
    return recnik

def vrati_recnik_1(reci):
    return {rec: len(rec) for rec in reci if rec}

print(vrati_recnik(['t', 'test', '', 'test123']))
print(vrati_recnik_1(['t', 'test', '', 'test123']))
#%% NUMPY - biblioteka za matrice
import numpy as np                   # uvozenje bibliteke za rad sa matricama

# INDEXING
vektor = np.array([1,2,4,5])         # inicijalizacija vektora (jednodimenzione matrice)
vektor*0.5                           # mnozenje vektora skalarom
vektor^2
vektor[2]                            # pristupanje vrednosti elementa vektora na poziciji 2
vektor[[1,3]]                        # uzimanje podniza, sastavljenog od elementa 1 i 3
matrica = np.array([[2,3,2],[3,4,5]]) # inicijalizacija matrice
matrica[0][2]                        # pristupanje elementa u matrici u redu 0 i koloni 2
matrica = np.zeros((10,2))           # inicijalizacija prazne matrice (elementi=0) sa 10 redova i 2 kolone

# ARITMETIKA MATRICA I ELEMENT-WISE I DELJENJA, TRANSPONOVANJE
m1 = np.random.rand(2,3)             # pravljenje matrice od slucajnih brojeva, u 2 reda i 3 kolone
m2 = np.random.rand(2,3)
m1.dot(m2)                           # skalarni proizvod matrice m1 i m2 (OVO NECE RADITI, jer dimenzije matrica se moraju slagati)
m1.dot(m2.T)                         # sklarni proizvod matrice m1 i transponovane matrice m2 (sada se dimenzije slazu za mnozenje matrica)
m1*m2                                # mnozenje elemenata dve matrice na istim pozicijama
m1-m2                                # oduzimanje elemenata dve matrice na istim pozicijama

# SUM, MEAN, MIN, ARGMIN
m1.sum(axis=1)                       # suma kolona matrice (rezultat: vektor-red)
m1.mean()                            # prosecne vrednosti u svakom redu matrice (rezultat: vektor-kolona)
m1.min(axis=0)                       # minimalne vrednosti u svakom redu
m2.argmin(axis=0)                    # indeksi kolone u kojoj je bila minimalna vrednost u svakom redu

# DELJENJE MATRICE SA NIZOM =>
# STANDARDIZACIJA MATRICE

#%% PANDAS - biblioteka za rad sa tabelama
import pandas as pd                  # uvozenje biblioteke

data = pd.DataFrame(m2, columns=['A','B','C'], index=['red1','red2'])  # kreiranje tabele na osnovu matrice m2, sa nazivima kolona i redova
print(data)

# SLICING... INDEX, LOC, ILOC
data['B']                            # izdvajanje kolone sa nazivom 'B'
data[['B','C']]                      # izdvajanje kolona sa nazivima 'B' i 'C'
data.loc['red1']                     # izdvajanje reda sa indeksom 'red1'
data.loc['red1','B']                 # izvlacenje vrednosti iz tabele u redu 'red1' i koloni 'B'
data.iloc[:,0]                       # izdvajanje svih redova u prvoj koloni
data.iloc[:,-1]                      # izdvajanje svih redova u poslednjoj koloni
data['D'] = data['A']+data['B']      # dodavanje nove kolone u tabelu, koja se racuna kao zbir 2 postojece kolone

# FILTERING []
data>0.5                             # ispitivanje svih vrednosti u tabeli da li zadovoljavaju uslov
data[data>0.5]                       # izvlacenje onih elemenata tabele koji zadovoljavaju uslov (filter)
data[data['A']>0.5]                  # izvlacenje onih redova matrice kod kojih je vrednost kolone 'A' veca od 0.5

# SUM, MEAN
data.mean()                          # izracunavanje prosecnih vrednosti u svakom redu tabele
data.sum(axis=1)                     # izracunavanje zbirova svake kolone

data.to_numpy()                      # pretvaranje tabele u matricu, za dalju matricni racun kroz biblioteku Numpy
data.values()
data.drop('A',axis=1)                # izbacivanje kolone 'A' iz tabele i vracanje redukovane tabele (ova funkcija ne modifikuje originalnu tabelu)

data = pd.read_csv('drug.csv')       # funkcija za ucitavanje podataka iz .csv fajla


#%% -------------------------------------- ZADATAK
# Iz podataka DRUG.CSV pronaci pacijenta koji ima najvecu kombinaciju natrijuma i kalijuma u odnosu (0.6 i 0.4)

# Koraci:
# ucitaj podatke o lekovima
import pandas as pd

data = pd.read_csv('data/drug.csv')

# uzmi samo numericke atribute (rucno i funkcijom isinstance)
numeric_att = []
for att in data.columns:
    if isinstance(data[att].iloc[0], (int, float)):
        numeric_att.append(att)
data_numeric = data[numeric_att]

# izracunaj otezanu sumu [0.4 0.6] 
score = 0.6 * data_numeric['Na'] + 0.4 * data_numeric['K']

# izaberi indeks elementa sa najvecom sumom
idx = score.idxmax()

# prikazi pacijenta sa tim indeksom
print(data.loc[idx])


#----------------------------------------------------
import pandas as pd

data = pd.read_csv('data/drug.csv')

score = 0.6 * data['Na'] + 0.4 * data['K']
print(data.loc[score.idxmax()])
