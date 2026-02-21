import pandas as pd
import numpy as np
import os

print("\n" + "="*80)
print("CÁLCULO DE MÉTRICAS - ESCENARIO 1 HORA")
print("="*80)

# Configuración de Archivos (1 Hora)
FILES = {
    'Lunes':     {'real': 'lunes_hora.csv',     'sim': 'Resultados_PPO_Lunes_1H_Final.csv'},
    'Martes':    {'real': 'martes_hora.csv',    'sim': 'Resultados_PPO_Martes_1H_Final.csv'},
    'Miércoles': {'real': 'miercoles_hora.csv', 'sim': 'Resultados_PPO_Miercoles_1H_Final.csv'},
    'Jueves':    {'real': 'jueves_hora.csv',    'sim': 'Resultados_PPO_Jueves_1H_Final.csv'},
    'Viernes':   {'real': 'viernes_hora.csv',   'sim': 'Resultados_PPO_Viernes_1H_Final.csv'}
}

# Encabezado de la Tabla
header = f"{'Día':<10} | {'CV(RMSE)':<10} | {'NMBE':<10} | {'RMSE':<8} | {'MAE':<8} | {'MAPE':<10} | {'Registros':<10}"
print(header)
print("-" * len(header))

for dia, paths in FILES.items():
    try:
        # 1. Cargar Real
        if not os.path.exists(paths['real']):
            print(f"{dia:<10} | {'FALTA ARCHIVO REAL':<50}")
            continue
        df_real = pd.read_csv(paths['real'])
        col_date = 'Local Time Stamp' if 'Local Time Stamp' in df_real.columns else 'Timestamp'
        df_real[col_date] = pd.to_datetime(df_real[col_date])
        df_real.set_index(col_date, inplace=True)

        # 2. Cargar Simulado
        if not os.path.exists(paths['sim']):
            print(f"{dia:<10} | {'FALTA ARCHIVO SIM':<50}")
            continue
        df_sim = pd.read_csv(paths['sim'])
        df_sim['Timestamp'] = pd.to_datetime(df_sim['Timestamp'])
        df_sim.set_index('Timestamp', inplace=True)

        # 3. Merge (Intersección exacta)
        df = pd.merge(df_real[['Act P (kW)']], df_sim[['Power_kW_PPO']], 
                      left_index=True, right_index=True, how='inner')

        # 4. Cálculos Matemáticos
        y_true = df['Act P (kW)'].values
        y_pred = df['Power_kW_PPO'].values
        n = len(df)

        if n == 0:
            print(f"{dia:<10} | {'SIN DATOS EN COMÚN':<50}")
            continue

        # RMSE & CV(RMSE)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        y_bar = np.mean(y_true)
        cv_rmse = (rmse / y_bar) * 100

        # NMBE (ASHRAE: n-1 en denominador)
        nmbe = (np.sum(y_pred - y_true) / ((n - 1) * y_bar)) * 100

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))

        # MAPE (Filtrando ceros)
        mask = y_true > 0.001
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

        # 5. Imprimir Fila Formateada
        print(f"{dia:<10} | {cv_rmse:>9.2f}% | {nmbe:>9.2f}% | {rmse:>8.2f} | {mae:>8.2f} | {mape:>9.2f}% | {n:>9,}")

    except Exception as e:
        print(f"{dia:<10} | ERROR: {str(e)}")

print("-" * len(header))
print(" Cálculo finalizado.\n")