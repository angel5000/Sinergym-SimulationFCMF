import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from datetime import datetime
import matplotlib.dates as mdates

print("GENERANDO GRÁFICAS COMPARATIVAS PARA MARTES...")

# --- 1. CARGA Y LIMPIEZA DE DATOS ---
try:
    # Datos Reales (Martes)
    df_real = pd.read_csv('martes_hora.csv')
    # Ajuste de nombre de columna de fecha si varía
    col_date = 'Local Time Stamp' if 'Local Time Stamp' in df_real.columns else 'Timestamp'
    df_real[col_date] = pd.to_datetime(df_real[col_date])
    df_real = df_real.sort_values(col_date).drop_duplicates(subset=[col_date])
    df_real = df_real.set_index(col_date)

    # Datos Simulados (Resultados Martes)
    df_sim = pd.read_csv('Resultados_PPO_Martes_1H_Final.csv')
    df_sim['Timestamp'] = pd.to_datetime(df_sim['Timestamp'])
    df_sim = df_sim.sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])
    df_sim = df_sim.set_index('Timestamp')

    print("Archivos cargados correctamente.")

except FileNotFoundError as e:
    print(f"Error: Falta el archivo {e.filename}")
    exit()

# --- 2. FUSIÓN DE DATOS ---
# Unimos por fecha exacta. Como el CSV simulado solo tiene Martes, 
# el resultado final será solo de Martes.
df_merged = pd.merge(df_real[['Act P (kW)']], df_sim[['Power_kW_PPO']], 
                     left_index=True, right_index=True, how='inner')

if df_merged.empty:
    print("Error: No hay fechas comunes. Verifica que 'martes_hora.csv' tenga fechas del 2021-2022.")
    exit()

# --- 3. CÁLCULO DE MÉTRICAS GLOBALES ---
rmse = np.sqrt(np.mean((df_merged['Act P (kW)'] - df_merged['Power_kW_PPO'])**2))
mae = np.mean(np.abs(df_merged['Act P (kW)'] - df_merged['Power_kW_PPO']))

# MAPE (Filtrando valores < 1 kW)
df_mape = df_merged[df_merged['Act P (kW)'] > 1.0]
mape = np.mean(np.abs((df_mape['Act P (kW)'] - df_mape['Power_kW_PPO']) / df_mape['Act P (kW)'])) * 100

# Texto para la tabla
stats_text = (f"Métricas Martes:\n"
              f"MAPE: {mape:.2f} %\n"
              f"RMSE: {rmse:.2f} kW\n"
              f"MAE:  {mae:.2f} kW")

print(f"Métricas Calculadas: MAPE={mape:.2f}%")

# Preparar datos extras
df_merged['year'] = df_merged.index.year
df_merged['hour_rounded'] = df_merged.index.hour

# Función de suavizado
def get_smooth_curve(x, y, n=300):
    if len(x) < 4: return x, y
    try:
        x_new = np.linspace(x.min(), x.max(), n)
        spl = make_interp_spline(x, y, k=3)
        y_new = spl(x_new)
        return x_new, np.clip(y_new, 0, None)
    except:
        return x, y

# Función para poner la tabla
def add_metrics_box(ax):
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black')
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontweight='bold', zorder=100)

# ==========================================
# GRÁFICA 1: PERFIL DIARIO PROMEDIO (MARTES)
# ==========================================
plt.figure(figsize=(12, 7))

hourly = df_merged.groupby('hour_rounded').mean().reset_index()
xr, yr = get_smooth_curve(hourly['hour_rounded'], hourly['Act P (kW)'])
xs, ys = get_smooth_curve(hourly['hour_rounded'], hourly['Power_kW_PPO'])

plt.plot(xr, yr, 'k--', linewidth=3, label='Real Promedio (Martes)')
plt.plot(xs, ys, 'g-', linewidth=3, label='IA Simulación (Martes)') # Verde para diferenciar
plt.fill_between(xr, yr, ys, color='gray', alpha=0.2, label='Error')

add_metrics_box(plt.gca()) 

plt.title('Perfil Diario Promedio (Martes) - Validación Final', fontsize=16, fontweight='bold')
plt.xlabel('Hora del Día')
plt.ylabel('Potencia (kW)')
plt.xticks(range(0, 25, 2))
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('Grafica1_Martes_Perfil.png', dpi=300)
print("Gráfica 1 lista.")
plt.close()

# ==========================================
# GRÁFICA 2: COMPARATIVA POR AÑO
# ==========================================
years = sorted(df_merged['year'].unique())
fig, axes = plt.subplots(len(years), 1, figsize=(12, 5 * len(years)))
if len(years) == 1: axes = [axes]

for i, year in enumerate(years):
    ax = axes[i]
    data_year = df_merged[df_merged['year'] == year]
    
    if not data_year.empty:
        hourly_y = data_year.groupby('hour_rounded').mean().reset_index()
        xr, yr = get_smooth_curve(hourly_y['hour_rounded'], hourly_y['Act P (kW)'])
        xs, ys = get_smooth_curve(hourly_y['hour_rounded'], hourly_y['Power_kW_PPO'])
        
        ax.plot(xr, yr, 'k--', linewidth=2, label=f'Real {year}')
        ax.plot(xs, ys, linewidth=3, label=f'IA {year}', color=f'C{i+2}') # Colores variados
        
        ax.set_title(f'Perfil Martes - Año {year}', fontweight='bold')
        ax.set_ylabel('kW')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_xticks(range(0, 25, 2))
        
        if i == 0: add_metrics_box(ax)

plt.xlabel('Hora')
plt.tight_layout()
plt.savefig('Grafica2_Martes_Anio.png', dpi=300)
print("Gráfica 2 lista.")
plt.close()

# ==========================================
# GRÁFICA 3: SERIE TEMPORAL (MARTES A MARTES)
# ==========================================
# Agrupamos por fecha exacta para evitar líneas en blanco
daily_points = df_merged.groupby(df_merged.index.date).mean()
dates = pd.to_datetime(daily_points.index)

plt.figure(figsize=(14, 7))

plt.plot(dates, daily_points['Act P (kW)'], 'k--o', alpha=0.6, label='Real (Promedio Martes)')
plt.plot(dates, daily_points['Power_kW_PPO'], 'g-o', linewidth=2, alpha=0.8, label='IA (Promedio Martes)')

add_metrics_box(plt.gca())

plt.title('Evolución Temporal: Comparación Martes a Martes', fontsize=16, fontweight='bold')
plt.ylabel('Potencia Promedio del Día (kW)')
plt.xlabel('Fecha')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig('Grafica3_Martes_Serie.png', dpi=300)
print("Gráfica 3 lista.")
plt.close()

print("\n¡TODO LISTO! Gráficas de Martes generadas.")
