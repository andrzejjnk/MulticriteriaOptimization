import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from math import pi
import pandas as pd
from io import BytesIO
from datetime import datetime
import os

# Przygotowanie algorytmów (przykładowe)
from alg_bez_filtr import find_non_dominated_points as alg_bez_filtr
from alg_filtr import find_non_dominated_points as alg_filtr
from alg_pkt_idealny import find_non_dominated_points as alg_pkt_idealny

# Mapowanie nazw algorytmów do funkcji
algorithms = {
    "algorytm bez filtracji": alg_bez_filtr,
    "algorytm z filtracja": alg_filtr,
    "algorytm punkt idealny": alg_pkt_idealny,
}


def save_to_excel_local(benchmark_df, criteria, data_df, batch_count, data_distribution, data_count, data_range, 
                        lambda_poisson, sigma_gauss, mu_gauss, lambda_exponential):
    # Ścieżka zapisu
    folder_path = "eksperymenty_excel"
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(folder_path, f"benchmark_results_{data_distribution}_{batch_count}_{timestamp}.xlsx")
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # Arkusz "Wyniki Benchmarku i Dane"
        start_row = 0
        
        # Zapis wyników benchmarku
        benchmark_df.to_excel(writer, index=False, startrow=start_row, sheet_name="Wyniki i Dane")
        start_row += len(benchmark_df) + 3  # Dodajemy odstęp po benchmarku
        
        # Zapis kryteriów z informacją o kierunku min/max
        criteria_df = pd.DataFrame({
            "Kryterium": [c[0] for c in criteria],
            "Min/Max": [c[1] for c in criteria]
        })
        criteria_df.to_excel(writer, index=False, startrow=start_row, sheet_name="Wyniki i Dane")
        start_row += len(criteria_df) + 3  # Dodajemy odstęp po kryteriach
        
        # Zapis ustawień benchmarku
        settings_df = pd.DataFrame({
            "Parametr": ["Liczba batchy", "Rozkład danych", "Liczba punktów", "Zakres wartości",
                         "λ Poisson", "σ Gauss", "μ Gauss", "λ Eksponencjalny"],
            "Wartość": [batch_count, data_distribution, data_count, str(data_range),
                        lambda_poisson, sigma_gauss, mu_gauss, lambda_exponential]
        })
        settings_df.to_excel(writer, index=False, startrow=start_row, sheet_name="Wyniki i Dane")
        start_row += len(settings_df) + 3  # Dodajemy odstęp po ustawieniach
        
        # Zapis pełnych danych wejściowych
        data_df.to_excel(writer, index=False, startrow=start_row, sheet_name="Wyniki i Dane")
        
    # Zapis danych do lokalnego pliku
    with open(file_path, "wb") as f:
        f.write(output.getvalue())

    # Przygotowanie pliku do pobrania
    output.seek(0)
    return file_path, output.getvalue()






# Ustawienia interfejsu i wczytywanie kryteriów
st.title("Optymalizacja punktów niezdominowanych")
st.write("Autorzy:")
st.write("Andrzej Janik oraz Artur Mazurkiewicz")
st.sidebar.header("Konfiguracja kryteriów")

# Panel edytora kryteriów
criteria = []
if "criteria_count" not in st.session_state:
    st.session_state.criteria_count = 2  # Domyślna liczba kryteriów

with st.sidebar.form("unique_criteria_form"):
    criteria_count = st.number_input("Ilość kryteriów", min_value=2, max_value=10, step=1, value=st.session_state.criteria_count)
    st.session_state.criteria_count = criteria_count  # Zapisujemy aktualną liczbę kryteriów
    for i in range(criteria_count):
        col1, col2 = st.columns([3, 2])
        with col1:
            criterion_name = st.text_input(f"Nazwa kryterium {i + 1}", value=f"Kryterium {i + 1}")
        with col2:
            direction = st.selectbox(f"Kierunek {i + 1}", ["Min", "Max"], index=0)
        criteria.append((criterion_name, direction))
    submit_button = st.form_submit_button(label="Zatwierdź kryteria")

# Generowanie losowego zbioru danych
st.sidebar.header("Generowanie danych")
data_config = st.sidebar.expander("Ustawienia danych")
with data_config:
    if "data_count" not in st.session_state:
        st.session_state.data_count = 10  # Domyślna liczba punktów
    if "data_distribution" not in st.session_state:
        st.session_state.data_distribution = "Jednostajny"  # Domyślny rozkład danych
    if "data_range" not in st.session_state:
        st.session_state.data_range = (10, 50)  # Domyślny zakres wartości
    
    data_count = st.number_input("Ilość punktów", min_value=1, max_value=1000, value=st.session_state.data_count)
    st.session_state.data_count = data_count  # Zapisujemy aktualną liczbę punktów
    
    data_distribution = st.selectbox("Rozkład danych", ["Jednostajny", "Gaussa", "Eksponencjonalny", "Poissona"], index=0)
    st.session_state.data_distribution = data_distribution  # Zapisujemy aktualny rozkład danych
    
    data_range = st.slider("Zakres wartości", min_value=0, max_value=100, value=st.session_state.data_range)
    st.session_state.data_range = data_range  # Zapisujemy aktualny zakres wartości

    lambda_poisson = st.number_input("Parametr λ (dla rozkładu Poissona)", min_value=0.1, max_value=10.0, value=2.0) if data_distribution == "Poissona" else None
    sigma_gauss = st.number_input("Odchylenie standardowe (dla rozkładu Gaussa)", min_value=0.1, max_value=20.0, value=10.0) if data_distribution == "Gaussa" else None
    mu_gauss = st.number_input("Wartość oczekiwana (dla rozkładu Gaussa)", min_value=0.0, max_value=100.0, value=30.0) if data_distribution == "Gaussa" else None
    lambda_exponential = st.number_input("Parametr λ (dla rozkładu eksponencjonalnego)", min_value=0.1, max_value=10.0, value=1.0) if data_distribution == "Eksponencjonalny" else None

generate_button = st.sidebar.button("Generuj losowe dane")

if generate_button:
    if data_distribution == "Jednostajny":
        data = np.random.uniform(data_range[0], data_range[1], (data_count, criteria_count))
    elif data_distribution == "Gaussa":
        data = np.random.normal(mu_gauss, sigma_gauss, (data_count, criteria_count))
    elif data_distribution == "Eksponencjonalny":
        data = np.random.exponential(1 / lambda_exponential, (data_count, criteria_count))
        data = data + data_range[0]
    elif data_distribution == "Poissona":
        data = np.random.poisson(lambda_poisson, (data_count, criteria_count))
    
    data = np.clip(data, data_range[0], data_range[1])
    df = pd.DataFrame(data, columns=[c[0] for c in criteria])
    st.session_state["data"] = df
else:
    df = st.session_state.get("data", pd.DataFrame(np.random.rand(10, criteria_count), columns=[f"Kryterium {i+1}" for i in range(criteria_count)]))
    st.session_state["data"] = df

# Wyświetlanie danych i opcje sortowania
st.subheader("Dane wejściowe")
sort_column = st.selectbox("Sortuj według kryterium", [c[0] for c in criteria])
sort_order = st.radio("Kolejność sortowania", ["Rosnąco", "Malejąco"])
if st.button("Sortuj"):
    df = df.sort_values(by=sort_column, ascending=(sort_order == "Rosnąco"))

st.dataframe(df)

# Wybór algorytmu i uruchomienie benchmarku
st.sidebar.header("Wybór algorytmu")
selected_algorithm_name = st.sidebar.selectbox("Algorytm", list(algorithms.keys()))
selected_algorithm = algorithms[selected_algorithm_name]

if st.button("Znajdź punkty niezdominowane"):
    points = df.values.tolist()
    
    # Rozpoczęcie pomiaru czasu
    start_time = time.time()
    
    # Wykonanie algorytmu
    non_dominated_points, num_comparisons = selected_algorithm(points)
    
    # Koniec pomiaru czasu
    execution_time = time.time() - start_time  # Czas w sekundach
    
    # Zapisywanie wyników do stanu sesji
    st.session_state["non_dominated_points"] = non_dominated_points
    st.session_state["num_comparisons"] = num_comparisons
    st.session_state["execution_time"] = execution_time  # Zapis czasu w sesji

# Wyświetlanie wyników, jeśli istnieją w stanie sesji
if "non_dominated_points" in st.session_state:
    non_dominated_points = st.session_state["non_dominated_points"]
    num_comparisons = st.session_state["num_comparisons"]
    execution_time = st.session_state["execution_time"]

    # Wyświetlanie liczby punktów niezdominowanych i formatowanie punktów
    st.subheader(f"Punkty niezdominowane (Liczba: {len(non_dominated_points)})")
    formatted_points = "\n".join([f"{i+1}: ({', '.join(map(str, p))})" for i, p in enumerate(non_dominated_points)])
    st.text(formatted_points)
    st.write(f"Liczba porównań: {num_comparisons}")
    
    # Wyświetlanie czasu wykonania
    st.write(f"Czas wykonania algorytmu: {execution_time:.4f} sekundy")

    # Wizualizacja wyników
    st.subheader("Wizualizacja wyników")

    if criteria_count == 2:
        fig, ax = plt.subplots()
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], label="Wszystkie punkty")
        ax.scatter([p[0] for p in non_dominated_points], [p[1] for p in non_dominated_points], color='red', label="Punkty niezdominowane")
        ax.set_xlabel(f"{criteria[0][0]} ({criteria[0][1]})")  # Dodanie kierunku
        ax.set_ylabel(f"{criteria[1][0]} ({criteria[1][1]})")  # Dodanie kierunku
        ax.legend()
        st.pyplot(fig)
    
    elif criteria_count == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], label="Wszystkie punkty")
        ax.scatter([p[0] for p in non_dominated_points], [p[1] for p in non_dominated_points], [p[2] for p in non_dominated_points], color='red', label="Punkty niezdominowane")
        ax.set_xlabel(f"{criteria[0][0]} ({criteria[0][1]})")
        ax.set_ylabel(f"{criteria[1][0]} ({criteria[1][1]})")
        ax.set_zlabel(f"{criteria[2][0]} ({criteria[2][1]})")
        ax.legend()
        st.pyplot(fig)
    
    else:
        vis_option = st.selectbox("Wybierz metodę wizualizacji dla N > 4", ["Redukcja wymiarów (PCA)", "Macierz punktów (pairplot)", "Wykres radarowy"])
        
        if vis_option == "Redukcja wymiarów (PCA)":
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(df)
            fig, ax = plt.subplots()
            ax.scatter(reduced_data[:, 0], reduced_data[:, 1], label="Wszystkie punkty")
            ax.scatter(pca.transform(non_dominated_points)[:, 0], pca.transform(non_dominated_points)[:, 1], color='red', label="Punkty niezdominowane")
            ax.set_xlabel("Składnik główny 1")
            ax.set_ylabel("Składnik główny 2")
            ax.legend()
            st.pyplot(fig)
        
        elif vis_option == "Macierz punktów (pairplot)":
            fig = sns.pairplot(df)
            st.pyplot(fig)
        
        elif vis_option == "Wykres radarowy":
            # Wykres radarowy dla punktów niezdominowanych
            num_vars = len(criteria)
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            angles += angles[:1]  # zamykanie wykresu

            fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            for point in non_dominated_points:
                values = point + point[:1]
                ax.plot(angles, values, linewidth=1, linestyle='solid')
                ax.fill(angles, values, alpha=0.25)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([c[0] for c in criteria])
            st.pyplot(fig)


# Dodanie przycisku benchmark z batchowaniem
st.sidebar.header("Benchmark z batchowaniem")
batch_count = st.sidebar.number_input("Liczba batchy", min_value=1, max_value=1000, value=50)
if st.sidebar.button("Benchmark"):
    benchmark_results = []
    
    # Wykonanie benchmarku dla wszystkich algorytmów
    for alg_name, alg_func in algorithms.items():
        total_comparisons = 0
        total_execution_time = 0
        
        for _ in range(batch_count):
            # Generowanie losowych danych na potrzeby batcha
            if data_distribution == "Jednostajny":
                batch_data = np.random.uniform(data_range[0], data_range[1], (data_count, criteria_count))
            elif data_distribution == "Gaussa":
                batch_data = np.random.normal(mu_gauss, sigma_gauss, (data_count, criteria_count))
            elif data_distribution == "Eksponencjonalny":
                batch_data = np.random.exponential(1 / lambda_exponential, (data_count, criteria_count))
                batch_data = batch_data + data_range[0]
            elif data_distribution == "Poissona":
                batch_data = np.random.poisson(lambda_poisson, (data_count, criteria_count))
            
            batch_data = np.clip(batch_data, data_range[0], data_range[1])
            points = batch_data.tolist()
            
            start_time = time.time()
            non_dominated_points, num_comparisons = alg_func(points)
            execution_time = time.time() - start_time
            
            total_comparisons += num_comparisons
            total_execution_time += execution_time
        
        # Dodanie wyników do listy
        benchmark_results.append({
            "Algorytm": alg_name,
            "Punkty niezdominowane": len(non_dominated_points),
            "Liczba porównań (średnia)": total_comparisons / batch_count,
            "Czas wykonania (s)": total_execution_time / batch_count
        })
    
    # Przekształcenie wyników do DataFrame i wyświetlenie tabeli
    benchmark_df = pd.DataFrame(benchmark_results)
    st.session_state["benchmark_df"] = benchmark_df
    st.subheader("Wyniki benchmarku")
    st.dataframe(benchmark_df)


if "benchmark_df" in st.session_state and not st.session_state["benchmark_df"].empty:
    # Zapisanie wyników benchmarku do pliku lokalnego i do pamięci
    file_path, excel_data = save_to_excel_local(
        st.session_state["benchmark_df"],
        criteria=criteria,
        data_df=st.session_state["data"],
        batch_count=batch_count,
        data_distribution=data_distribution,
        data_count=data_count,
        data_range=data_range,
        lambda_poisson=lambda_poisson,
        sigma_gauss=sigma_gauss,
        mu_gauss=mu_gauss,
        lambda_exponential=lambda_exponential
    )
    
    # Dodanie przycisku pobierania
    st.download_button(
        label="Pobierz wyniki jako Excel",
        data=excel_data,
        file_name=os.path.basename(file_path),
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )