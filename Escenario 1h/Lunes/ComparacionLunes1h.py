import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from datetime import datetime
import matplotlib.dates as mdates
import os

print("GENERANDO GRAFICOS DE COMPARACION PARA EL LUNES...")

# --- 1. CARGA Y LIMPIEZA DE DATOS ---
try:
    real_file = 'lunes_hora.csv'
    if not os.path.exists(real_file) and os.path.exists('Lunes_Hora.csv'):
        real_file = 'Lunes_Hora.csv'
    df_real = pd.read_csv(real_file)
    col_date = 'Local Time Stamp' if 'Local Time Stamp' in df_real.columns else 'Timestamp'
    df_real[col_date] = pd.to_datetime(df_real[col_date])
    df_real = df_real.sort_values(col_date).drop_duplicates(subset=[col_date])
    df_real = df_real.set_index(col_date)
    
    sim_file = 'Resultados_PPO_Lunes_1H_Final.csv'
    df_sim = pd.read_csv(sim_file)
    df_sim['Timestamp'] = pd.to_datetime(df_sim['Timestamp'])
    df_sim = df_sim.sort_values('Timestamp').drop_duplicates(subset=['Timestamp'])
    df_sim = df_sim.set_index('Timestamp')
    
    print(f"Archivos cargados: {real_file} y {sim_file}")
except FileNotFoundError as e:
    print(f"Error: Archivo no encontrado {e.filename}")
    exit()

# --- 2. UNION DE DATOS ---
df_merged = pd.merge(df_real[['Act P (kW)']], df_sim[['Power_kW_PPO']],
                     left_index=True, right_index=True, how='inner')

if df_merged.empty:
    print("Error: No hay fechas en comun. Revisar los anios de los CSV.")
    exit()

# --- 3. CALCULO DE METRICAS GLOBALES ---
rmse = np.sqrt(np.mean((df_merged['Act P (kW)'] - df_merged['Power_kW_PPO'])**2))
mae = np.mean(np.abs(df_merged['Act P (kW)'] - df_merged['Power_kW_PPO']))
df_mape = df_merged[df_merged['Act P (kW)'] > 1.0]
mape = np.mean(np.abs((df_mape['Act P (kW)'] - df_mape['Power_kW_PPO']) / df_mape['Act P (kW)'])) * 100

stats_text = (f"Metricas Lunes:\n"
              f"MAPE: {mape:.2f} %\n"
              f"RMSE: {rmse:.2f} kW\n"
              f"MAE: {mae:.2f} kW")

print(f"Metricas calculadas: MAPE={mape:.2f}%")

df_merged['year'] = df_merged.index.year
df_merged['hour_rounded'] = df_merged.index.hour

# Funcion de suavizado por spline cubico
def get_smooth_curve(x, y, n=300):
    if len(x) < 4: return x, y
    try:
        sorted_idx = np.argsort(x)
        x = np.array(x)[sorted_idx]
        y = np.array(y)[sorted_idx]
        x_new = np.linspace(x.min(), x.max(), n)
        spl = make_interp_spline(x, y, k=3)
        y_new = spl(x_new)
        return x_new, np.clip(y_new, 0, None)
    except:
        return x, y

# Funcion para agregar cuadro de metricas en el grafico
def add_metrics_box(ax):
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold', zorder=100)

# ==========================================
# GRAFICO 1: PERFIL DIARIO PROMEDIO
# ==========================================
plt.figure(figsize=(14, 8))
hourly = df_merged.groupby('hour_rounded').mean().reset_index()
xr, yr = get_smooth_curve(hourly['hour_rounded'], hourly['Act P (kW)'])
xs, ys = get_smooth_curve(hourly['hour_rounded'], hourly['Power_kW_PPO'])

plt.plot(xr, yr, 'k--', linewidth=3, label='Real (Avg)', zorder=3)
plt.plot(xs, ys, color='#d62728', linewidth=3.5, label='AI (PPO Simulation)', zorder=4)

margin = np.abs(yr - ys)
if len(xr) == len(xs):
    plt.fill_between(xr, yr - margin/2, yr + margin/2, alpha=0.15, color='gray', label='Error Margin')

add_metrics_box(plt.gca())

plt.title('Average Daily Profile - Monday (1H)', fontsize=16, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=14, fontweight='bold')
plt.ylabel('Active Power (kW)', fontsize=14, fontweight='bold')
plt.xticks(range(0, 25, 2))
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', shadow=True, fontsize=12)
plt.tight_layout()
plt.savefig('model_validation_daily_profile_monday_1h.png', dpi=300)
print("Grafico 1 generado.")
plt.close()

# ==========================================
# GRAFICO 2: COMPARACION POR ANIO
# ==========================================
years = sorted(df_merged['year'].unique())
fig, axes = plt.subplots(len(years), 1, figsize=(14, 6 * len(years)))
if len(years) == 1: axes = [axes]

colors_year = {2021: '#1f77b4', 2022: '#ff7f0e', 2023: '#2ca02c'}

for i, year in enumerate(years):
    ax = axes[i]
    data_year = df_merged[df_merged['year'] == year]
    if not data_year.empty:
        hourly_y = data_year.groupby('hour_rounded').mean().reset_index()
        xr, yr = get_smooth_curve(hourly_y['hour_rounded'], hourly_y['Act P (kW)'])
        xs, ys = get_smooth_curve(hourly_y['hour_rounded'], hourly_y['Power_kW_PPO'])
        
        c = colors_year.get(year, 'blue')
        ax.plot(xr, yr, 'k--', linewidth=2.5, label=f'Real {year}', alpha=0.8)
        ax.plot(xs, ys, color=c, linewidth=3, label=f'PPO {year}', alpha=0.9)
        ax.set_title(f'Monday Profile - Year {year}', fontweight='bold', fontsize=14)
        ax.set_ylabel('Power (kW)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', shadow=True)
        ax.set_xticks(range(0, 25, 2))
        
        if i == 0: add_metrics_box(ax)

plt.xlabel('Hour of Day', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('model_validation_yearly_monday_1h.png', dpi=300)
print("Grafico 2 generado.")
plt.close()

# ==========================================
# GRAFICO 3: SERIE TEMPORAL COMPLETA
# ==========================================
daily_points = df_merged.groupby(df_merged.index.date).mean()
dates = pd.to_datetime(daily_points.index)

plt.figure(figsize=(18, 9))
x_days = np.arange(len(dates))

if len(x_days) > 10:
    xr, yr = get_smooth_curve(x_days, daily_points['Act P (kW)'].values, n=len(x_days)*3)
    xs, ys = get_smooth_curve(x_days, daily_points['Power_kW_PPO'].values, n=len(x_days)*3)
    date_range_smooth = pd.date_range(start=dates[0], end=dates[-1], periods=len(xr))
    plt.plot(date_range_smooth, yr, 'k--', linewidth=2, alpha=0.8, label='Real (Avg)')
    plt.plot(date_range_smooth, ys, 'b-', linewidth=2.5, alpha=0.9, label='PPO (Simulation)')
else:
    plt.plot(dates, daily_points['Act P (kW)'], 'k--o', label='Real')
    plt.plot(dates, daily_points['Power_kW_PPO'], 'b-o', label='PPO')

add_metrics_box(plt.gca())

plt.title('Model Validation - Full Time Series - Monday 1H', fontsize=16, fontweight='bold')
plt.ylabel('Daily Avg Power (kW)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', shadow=True, fontsize=12)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig('model_validation_time_series_monday_1h.png', dpi=300)
print("Grafico 3 generado.")
plt.close()

print("\nTODO LISTO. Graficos del lunes generados con tablas de error.")
