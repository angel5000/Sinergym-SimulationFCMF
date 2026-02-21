import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from datetime import datetime
import matplotlib.dates as mdates
import os

print("GENERANDO GRAFICAS COMPARATIVAS MIERCOLES (15 MIN)...")

# --- 1. CARGA Y LIMPIEZA DE DATOS ---
try:
    # 1. Datos Reales
    real_file = 'Miercoles.csv'
    if not os.path.exists(real_file): real_file = 'miercoles.csv'
    
    df_real = pd.read_csv(real_file)
    col_date = 'Local Time Stamp' if 'Local Time Stamp' in df_real.columns else 'Timestamp'
    df_real[col_date] = pd.to_datetime(df_real[col_date])
    df_real = df_real.sort_values(col_date).drop_duplicates(subset=[col_date])
    df_real = df_real.set_index(col_date)

    # 2. Datos Simulados (Resultados 15 Min)
    sim_file = 'Resultados_PPO_Miercoles_15Min_Final.csv' 
    df_sim = pd.read_csv(sim_file)
    df_sim['Timestamp'] = pd.to_datetime(df_sim['Timestamp'])
    df_sim = df_sim.sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])
    df_sim = df_sim.set_index('Timestamp')

    print(f"Archivos cargados: {real_file} y {sim_file}")

except FileNotFoundError as e:
    print(f"Error: Falta el archivo {e.filename}")
    exit()

# --- 2. FUSIÓN DE DATOS ---
df_merged = pd.merge(df_real[['Act P (kW)']], df_sim[['Power_kW_PPO']], 
                     left_index=True, right_index=True, how='inner')

if df_merged.empty:
    print("Error: No hay fechas comunes.")
    exit()

# --- 3. MÉTRICAS ---
rmse = np.sqrt(np.mean((df_merged['Act P (kW)'] - df_merged['Power_kW_PPO'])**2))
mae = np.mean(np.abs(df_merged['Act P (kW)'] - df_merged['Power_kW_PPO']))
df_mape = df_merged[df_merged['Act P (kW)'] > 0.1]
mape = np.mean(np.abs((df_mape['Act P (kW)'] - df_mape['Power_kW_PPO']) / df_mape['Act P (kW)'])) * 100

stats_text = (f"Métricas Miércoles (15 min):\n"
              f"MAPE: {mape:.2f} %\n"
              f"RMSE: {rmse:.2f} kW\n"
              f"MAE:  {mae:.2f} kW")

print(f"Metricas: MAPE={mape:.2f}%")

# Preparar hora decimal para 15 min (ej: 10:30 -> 10.5)
df_merged['year'] = df_merged.index.year
df_merged['hour_decimal'] = df_merged.index.hour + (df_merged.index.minute / 60.0)

# Función de suavizado
def get_smooth_curve(x, y, n=500):
    if len(x) < 4: return x, y
    try:
        sorted_idx = np.argsort(x)
        x = np.array(x)[sorted_idx]
        y = np.array(y)[sorted_idx]
        x_uniq, idx_uniq = np.unique(x, return_index=True)
        y_uniq = y[idx_uniq]
        if len(x_uniq) < 4: return x_uniq, y_uniq
        x_new = np.linspace(x_uniq.min(), x_uniq.max(), n)
        spl = make_interp_spline(x_uniq, y_uniq, k=3)
        return x_new, np.clip(spl(x_new), 0, None)
    except:
        return x, y

def add_metrics_box(ax):
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold', zorder=100)

# ==========================================
# GRÁFICA 1: PERFIL DIARIO
# ==========================================
plt.figure(figsize=(14, 8))
hourly = df_merged.groupby('hour_decimal').mean().reset_index()
xr, yr = get_smooth_curve(hourly['hour_decimal'], hourly['Act P (kW)'])
xs, ys = get_smooth_curve(hourly['hour_decimal'], hourly['Power_kW_PPO'])

plt.plot(xr, yr, 'k--', linewidth=3, label='Real (Medio)', zorder=3)
plt.plot(xs, ys, color='#d62728', linewidth=3.5, label='IA (Simulación PPO)', zorder=4)

margin = np.abs(yr - ys)
if len(xr) == len(xs):
    plt.fill_between(xr, yr - margin/2, yr + margin/2, alpha=0.15, color='gray', label='Margen de Error')

add_metrics_box(plt.gca())
plt.title('Perfil Diario Promedio - Miércoles (Resolución 15 Min)', fontsize=16, fontweight='bold')
plt.xlabel('Hora del Día', fontsize=14, fontweight='bold')
plt.ylabel('Potencia Activa (kW)', fontsize=14, fontweight='bold')
plt.xticks(range(0, 25, 2))
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', shadow=True, fontsize=12)
plt.tight_layout()
plt.savefig('validacion_modelo_perfil_diario_miercoles_15min.png', dpi=300)
print("Grafica 1 lista.")
plt.close()

# ==========================================
# GRÁFICA 2: COMPARATIVA POR AÑO
# ==========================================
years = sorted(df_merged['year'].unique())
fig, axes = plt.subplots(len(years), 1, figsize=(14, 6 * len(years)))
if len(years) == 1: axes = [axes]
colors = {2021: '#1f77b4', 2022: '#ff7f0e', 2023: '#2ca02c'}

for i, year in enumerate(years):
    ax = axes[i]
    data_year = df_merged[df_merged['year'] == year]
    if not data_year.empty:
        h_y = data_year.groupby('hour_decimal').mean().reset_index()
        xr, yr = get_smooth_curve(h_y['hour_decimal'], h_y['Act P (kW)'])
        xs, ys = get_smooth_curve(h_y['hour_decimal'], h_y['Power_kW_PPO'])
        
        ax.plot(xr, yr, 'k--', linewidth=2.5, label=f'Real {year}', alpha=0.8)
        ax.plot(xs, ys, color=colors.get(year, 'blue'), linewidth=3, label=f'PPO {year}', alpha=0.9)
        ax.set_title(f'Perfil Miércoles - Año {year} (15 min)', fontweight='bold', fontsize=14)
        ax.set_ylabel('Potencia (kW)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', shadow=True)
        ax.set_xticks(range(0, 25, 2))
        if i == 0: add_metrics_box(ax)

plt.xlabel('Hora del Día', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('validacion_modelo_por_anio_miercoles_15min.png', dpi=300)
print("Grafica 2 lista.")
plt.close()

# ==========================================
# GRÁFICA 3: SERIE TEMPORAL
# ==========================================
daily_points = df_merged.groupby(df_merged.index.date).mean()
dates = pd.to_datetime(daily_points.index)

plt.figure(figsize=(18, 9))
x_days = np.arange(len(dates))
if len(x_days) > 10:
    xr, yr = get_smooth_curve(x_days, daily_points['Act P (kW)'].values, n=len(x_days)*3)
    xs, ys = get_smooth_curve(x_days, daily_points['Power_kW_PPO'].values, n=len(x_days)*3)
    d_smooth = pd.date_range(start=dates[0], end=dates[-1], periods=len(xr))
    plt.plot(d_smooth, yr, 'k--', linewidth=2, alpha=0.8, label='Real (Promedio)')
    plt.plot(d_smooth, ys, 'b-', linewidth=2.5, alpha=0.9, label='PPO (Simulación)')
else:
    plt.plot(dates, daily_points['Act P (kW)'], 'k--o', label='Real')
    plt.plot(dates, daily_points['Power_kW_PPO'], 'b-o', label='PPO')

add_metrics_box(plt.gca())
plt.title('Validación del Modelo - Serie Temporal Completa - Miércoles (15 Min)', fontsize=16, fontweight='bold')
plt.ylabel('Potencia Promedio del Día (kW)', fontsize=14, fontweight='bold')
plt.xlabel('Fecha', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', shadow=True, fontsize=12)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig('validacion_modelo_serie_temporal_miercoles_15min.png', dpi=300)
print("Grafica 3 lista.")
plt.close()