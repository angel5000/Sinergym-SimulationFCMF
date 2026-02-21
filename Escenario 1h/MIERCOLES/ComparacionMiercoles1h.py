import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from datetime import datetime
import matplotlib.dates as mdates
import os

print("GENERANDO GRÁFICAS COMPARATIVAS PARA MIÉRCOLES...")

# --- 1. CARGA Y LIMPIEZA DE DATOS ---
try:
    # Intentar cargar archivo real (Manejo de mayúsculas/minúsculas)
    real_file = 'miercoles_hora.csv'
    if not os.path.exists(real_file) and os.path.exists('Miercoles_Hora.csv'):
        real_file = 'Miercoles_Hora.csv'
        
    df_real = pd.read_csv(real_file)
    # Ajuste de nombre de columna de fecha
    col_date = 'Local Time Stamp' if 'Local Time Stamp' in df_real.columns else 'Timestamp'
    df_real[col_date] = pd.to_datetime(df_real[col_date])
    df_real = df_real.sort_values(col_date).drop_duplicates(subset=[col_date])
    df_real = df_real.set_index(col_date)

    # Cargar datos simulados (El archivo final generado)
    sim_file = 'Resultados_PPO_Miercoles_1H_Final.csv'
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
    print("Error: No hay fechas comunes. Verifica los años de los CSV.")
    exit()

# --- 3. CÁLCULO DE MÉTRICAS GLOBALES ---
rmse = np.sqrt(np.mean((df_merged['Act P (kW)'] - df_merged['Power_kW_PPO'])**2))
mae = np.mean(np.abs(df_merged['Act P (kW)'] - df_merged['Power_kW_PPO']))

# MAPE (Filtrando valores < 1 kW para evitar errores matemáticos)
df_mape = df_merged[df_merged['Act P (kW)'] > 1.0]
mape = np.mean(np.abs((df_mape['Act P (kW)'] - df_mape['Power_kW_PPO']) / df_mape['Act P (kW)'])) * 100

# Texto para la tabla
stats_text = (f"Métricas Miércoles:\n"
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
        # Ordenar y limpiar para spline
        sorted_idx = np.argsort(x)
        x = np.array(x)[sorted_idx]
        y = np.array(y)[sorted_idx]
        x_new = np.linspace(x.min(), x.max(), n)
        spl = make_interp_spline(x, y, k=3)
        y_new = spl(x_new)
        return x_new, np.clip(y_new, 0, None)
    except:
        return x, y

# Función para poner la tabla (Caja de Errores)
def add_metrics_box(ax):
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black', linewidth=2)
    ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontweight='bold', zorder=100)

# ==========================================
# GRÁFICA 1: PERFIL DIARIO PROMEDIO
# ==========================================
plt.figure(figsize=(14, 8))

hourly = df_merged.groupby('hour_rounded').mean().reset_index()
xr, yr = get_smooth_curve(hourly['hour_rounded'], hourly['Act P (kW)'])
xs, ys = get_smooth_curve(hourly['hour_rounded'], hourly['Power_kW_PPO'])

plt.plot(xr, yr, 'k--', linewidth=3, label='Real (Medio)', zorder=3)
plt.plot(xs, ys, color='#d62728', linewidth=3.5, label='IA (Simulación PPO)', zorder=4)

margin = np.abs(yr - ys)
# Rellenar solo si las longitudes coinciden (por el suavizado)
if len(xr) == len(xs):
    plt.fill_between(xr, yr - margin/2, yr + margin/2, alpha=0.15, color='gray', label='Margen de Error')

add_metrics_box(plt.gca()) # <--- TABLA

plt.title('Perfil Diario Promedio - Miércoles (1H)', fontsize=16, fontweight='bold')
plt.xlabel('Hora del Día', fontsize=14, fontweight='bold')
plt.ylabel('Potencia Activa (kW)', fontsize=14, fontweight='bold')
plt.xticks(range(0, 25, 2))
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', shadow=True, fontsize=12)
plt.tight_layout()
    plt.savefig('validacion_modelo_perfil_diario_miercoles_1h.png', dpi=300)
    print("Gráfica 1 lista.")
plt.close()

# ==========================================
# GRÁFICA 2: COMPARATIVA POR AÑO
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
        
        ax.set_title(f'Perfil Miércoles - Año {year}', fontweight='bold', fontsize=14)
        ax.set_ylabel('Potencia (kW)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', shadow=True)
        ax.set_xticks(range(0, 25, 2))
        
        # Agregar tabla al primer gráfico (o a todos si prefieres, aquí está en el primero)
        if i == 0: add_metrics_box(ax)

plt.xlabel('Hora del Día', fontsize=14, fontweight='bold')
plt.tight_layout()
    plt.savefig('validacion_modelo_por_anio_miercoles_1h.png', dpi=300)
    print("Gráfica 2 lista.")
plt.close()

# ==========================================
# GRÁFICA 3: SERIE TEMPORAL (MIÉRCOLES A MIÉRCOLES)
# ==========================================
# Agrupamos por fecha exacta para evitar líneas en blanco
daily_points = df_merged.groupby(df_merged.index.date).mean()
dates = pd.to_datetime(daily_points.index)

plt.figure(figsize=(18, 9))

# Usamos curva suave para la serie temporal también si hay muchos puntos
x_days = np.arange(len(dates))
if len(x_days) > 10:
    xr, yr = get_smooth_curve(x_days, daily_points['Act P (kW)'].values, n=len(x_days)*3)
    xs, ys = get_smooth_curve(x_days, daily_points['Power_kW_PPO'].values, n=len(x_days)*3)
    
    # Reconstruir fechas para el eje X
    date_range_smooth = pd.date_range(start=dates[0], end=dates[-1], periods=len(xr))
    
    plt.plot(date_range_smooth, yr, 'k--', linewidth=2, alpha=0.8, label='Real (Promedio)')
    plt.plot(date_range_smooth, ys, 'b-', linewidth=2.5, alpha=0.9, label='PPO (Simulación)')
else:
    plt.plot(dates, daily_points['Act P (kW)'], 'k--o', label='Real')
    plt.plot(dates, daily_points['Power_kW_PPO'], 'b-o', label='PPO')

add_metrics_box(plt.gca()) # <--- TABLA

plt.title('Validación del Modelo - Serie Temporal Completa - Miércoles 1H', fontsize=16, fontweight='bold')
plt.ylabel('Potencia Promedio del Día (kW)', fontsize=14, fontweight='bold')
plt.xlabel('Fecha', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', shadow=True, fontsize=12)

# Formato Fechas
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.gcf().autofmt_xdate()

plt.tight_layout()
    plt.savefig('validacion_modelo_serie_temporal_miercoles_1h.png', dpi=300)
    print("Gráfica 3 lista.")
plt.close()

print("\n¡TODO LISTO! Gráficas de Miércoles generadas con tablas de error.")
