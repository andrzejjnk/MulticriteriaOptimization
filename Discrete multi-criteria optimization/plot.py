import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Wczytanie danych z pliku Excel
# file_path = 'K1.xlsx'  # Podaj ścieżkę do swojego pliku
# file_path = 'K2.xlsx'  # Podaj ścieżkę do swojego pliku
data = pd.read_excel(file_path)

# Podział na dane o różnych rozkładach
rozkład_jednostajny = data[data['Rozkład'].str.contains("Jednostajny")]
rozkład_gaussa = data[data['Rozkład'].str.contains("Gaussa")]

# Funkcja rysująca wykres dla danego rozkładu z liniami regresji
def rysuj_wykres(dane, tytuł, plik_wyjsciowy):
    algorytmy = dane['Algorytm'].unique()
    kolory = ['red', 'green', 'blue']  # Kolory dla algorytmów
    
    plt.figure(figsize=(10, 6))
    
    for i, algorytm in enumerate(algorytmy):
        dane_alg = dane[dane['Algorytm'] == algorytm]
        x = dane_alg['Liczba kryteriów']
        y = dane_alg['Punkty niezdominowane']
        
        # Rysowanie punktów
        plt.scatter(x, y, label=algorytm, color=kolory[i], s=50)
        
        # Obliczenie i rysowanie linii regresji
        if len(x) > 1:  # Tylko jeśli mamy wystarczającą ilość punktów do regresji
            slope, intercept, _, _, _ = linregress(x, y)
            plt.plot(x, slope * x + intercept, color=kolory[i], linestyle='--', label=f"Linia regresji: {algorytm}")
    
    plt.title(tytuł)
    plt.xlabel('Liczba kryteriów')
    plt.ylabel('Liczba punktów niezdominowanych')
    plt.legend(title="Algorytmy i linie regresji")
    plt.grid(True)
    
    # Zapisanie wykresu do pliku
    plt.savefig(plik_wyjsciowy)
    plt.show()

# Rysowanie wykresów dla obu rozkładów
# rysuj_wykres(rozkład_jednostajny, 'Punkty niezdominowane w zależności o ilości kryteriów dla rozkładu Jednostajny na [0, 2]', 'K1_rozkład_jednostajny.png')
# rysuj_wykres(rozkład_gaussa, 'Punkty niezdominowane w zależności o ilości kryteriów dla rozkładu Gaussa (1,1)', 'K1_rozkład_gaussa.png')
# rysuj_wykres(rozkład_jednostajny, 'Punkty niezdominowane w zależności o ilości kryteriów dla rozkładu Jednostajny na [0, 2]', 'K2_rozkład_jednostajny.png')
# rysuj_wykres(rozkład_gaussa, 'Punkty niezdominowane w zależności o ilości kryteriów dla rozkładu Gaussa (1,1)', 'K2_rozkład_gaussa.png')
