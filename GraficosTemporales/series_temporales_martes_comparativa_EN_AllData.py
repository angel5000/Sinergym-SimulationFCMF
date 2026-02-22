import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend (avoids Tcl/Tk error)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

# Configure style for academic graphs
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# Base path (SIMULACIONES folder)
base_path = os.path.dirname(os.path.abspath(__file__))
sim_path = os.path.join(os.path.dirname(base_path), 'SIMULACIONES')

# ============================================================================
# LOAD TUESDAY DATA
# ============================================================================

# 15-minute data
df_real_15min = pd.read_csv(os.path.join(sim_path, '15minutos/martes/Martes.csv'))
df_real_15min['Potencia_kW'] = df_real_15min['Act P (kW)']
df_real_15min['Fecha'] = pd.to_datetime(df_real_15min['Local Time Stamp'])

df_ppo_15min = pd.read_csv(os.path.join(sim_path, '15minutos/martes/Resultados_PPO_Martes_15Min_Final.csv'))
df_ppo_15min['PPO_Potencia_kW'] = df_ppo_15min['Power_kW_PPO']
df_ppo_15min['Fecha'] = pd.to_datetime(df_ppo_15min['Timestamp'])

# 1-hour data
df_real_1h = pd.read_csv(os.path.join(sim_path, 'semana 1h/martes/martes_hora.csv'))
df_real_1h['Potencia_kW'] = df_real_1h['Act P (kW)']
df_real_1h['Fecha'] = pd.to_datetime(df_real_1h['Local Time Stamp'])

df_ppo_1h = pd.read_csv(os.path.join(sim_path, 'semana 1h/martes/Resultados_PPO_Martes_1H_Final.csv'))
df_ppo_1h['PPO_Potencia_kW'] = df_ppo_1h['Power_kW_PPO']
df_ppo_1h['Fecha'] = pd.to_datetime(df_ppo_1h['Timestamp'])

print(f"Data loaded:")
print(f"  15-min real: {df_real_15min['Fecha'].min()} to {df_real_15min['Fecha'].max()} ({len(df_real_15min)} rows)")
print(f"  15-min sim:  {df_ppo_15min['Fecha'].min()} to {df_ppo_15min['Fecha'].max()} ({len(df_ppo_15min)} rows)")
print(f"  1h real:     {df_real_1h['Fecha'].min()} to {df_real_1h['Fecha'].max()} ({len(df_real_1h)} rows)")
print(f"  1h sim:      {df_ppo_1h['Fecha'].min()} to {df_ppo_1h['Fecha'].max()} ({len(df_ppo_1h)} rows)")

# ============================================================================
# HELPER: Calculate ASHRAE G14 metrics
# ============================================================================
def calc_metrics(real, sim):
    """Calculate ASHRAE G14 validation metrics"""
    n = len(real)
    mean_real = real.mean()
    
    # CV(RMSE)
    rmse = np.sqrt(np.mean((real - sim)**2))
    cvrmse = (rmse / mean_real) * 100 if mean_real != 0 else 0
    
    # NMBE
    nmbe = (np.sum(sim - real) / (n * mean_real)) * 100 if mean_real != 0 else 0
    
    # MAE
    mae = np.mean(np.abs(real - sim))
    
    # MAPE (filter near-zero values)
    mask = real > 0.1
    if mask.sum() > 0:
        mape = np.mean(np.abs((real[mask] - sim[mask]) / real[mask])) * 100
    else:
        mape = 0
    
    return cvrmse, nmbe, rmse, mae, mape, n

# ============================================================================
# CHART: COMPLETE TIME SERIES - TUESDAY (15 MIN VS 1 HOUR) — ALL DATA
# ============================================================================

# Merge for 15 minutes
merged_15min = pd.merge(df_real_15min[['Fecha', 'Potencia_kW']], 
                         df_ppo_15min[['Fecha', 'PPO_Potencia_kW']], 
                         on='Fecha', how='inner')

# Merge for 1 hour
merged_1h = pd.merge(df_real_1h[['Fecha', 'Potencia_kW']], 
                      df_ppo_1h[['Fecha', 'PPO_Potencia_kW']], 
                      on='Fecha', how='inner')

print(f"\nMerged data:")
print(f"  15-min: {len(merged_15min)} common points")
print(f"  1h:     {len(merged_1h)} common points")

# Calculate metrics dynamically
cvrmse_15, nmbe_15, rmse_15, mae_15, mape_15, n_15 = calc_metrics(
    merged_15min['Potencia_kW'].values, merged_15min['PPO_Potencia_kW'].values)

cvrmse_1h, nmbe_1h, rmse_1h, mae_1h, mape_1h, n_1h = calc_metrics(
    merged_1h['Potencia_kW'].values, merged_1h['PPO_Potencia_kW'].values)

print(f"\nMetrics calculated:")
print(f"  15-min: CV(RMSE)={cvrmse_15:.2f}%, NMBE={nmbe_15:.2f}%, MAPE={mape_15:.2f}%")
print(f"  1h:     CV(RMSE)={cvrmse_1h:.2f}%, NMBE={nmbe_1h:.2f}%, MAPE={mape_1h:.2f}%")

# Weekly averages (more points for better curve visualization)
weekly_15min_real = merged_15min.groupby([merged_15min['Fecha'].dt.to_period('W')])['Potencia_kW'].mean()
weekly_15min_ppo = merged_15min.groupby([merged_15min['Fecha'].dt.to_period('W')])['PPO_Potencia_kW'].mean()
weekly_1h_real = merged_1h.groupby([merged_1h['Fecha'].dt.to_period('W')])['Potencia_kW'].mean()
weekly_1h_ppo = merged_1h.groupby([merged_1h['Fecha'].dt.to_period('W')])['PPO_Potencia_kW'].mean()

# NO YEAR FILTER - use all available data

# Determine the actual year range from data
all_years_15min = sorted(set(merged_15min['Fecha'].dt.year))
all_years_1h = sorted(set(merged_1h['Fecha'].dt.year))
year_range_15min = f"{min(all_years_15min)}-{max(all_years_15min)}"
year_range_1h = f"{min(all_years_1h)}-{max(all_years_1h)}"

# Color mapping for each year
year_colors = {2021: 'blue', 2022: 'orange', 2023: 'green', 2024: 'red', 2025: 'purple'}

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# --- Subplot 1: 15 minutes ---
fechas_15min = [p.to_timestamp() for p in weekly_15min_real.index]
valores_real_15min = weekly_15min_real.values
valores_ppo_15min = weekly_15min_ppo.values

for year in all_years_15min:
    idx = [i for i, f in enumerate(fechas_15min) if f.year == year]
    if idx:
        c = year_colors.get(year, 'gray')
        fechas_year = [fechas_15min[i] for i in idx]
        ax1.plot(fechas_year, valores_real_15min[idx], color=c, linestyle='--', 
                 linewidth=1.5, marker='o', markersize=3, label=f'Actual {year}', alpha=0.8)
        ax1.plot(fechas_year, valores_ppo_15min[idx], color=c, linewidth=2, 
                 marker='s', markersize=3, label=f'Simulated {year}')

ax1.set_ylabel('Average Power (kW)', fontweight='bold')
ax1.set_title(f'Interval: 15 minutes ({year_range_15min})', fontweight='bold', fontsize=11)
ax1.legend(loc='upper right', ncol=2, frameon=True, fancybox=True, shadow=True)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Add dynamically calculated metrics
textstr_15min = (f'VALIDATION (ASHRAE G14)\n'
                 f'CV(RMSE): {cvrmse_15:.2f} %\n'
                 f'NMBE: {nmbe_15:.2f} %\n'
                 f'RMSE: {rmse_15:.2f} kW\n'
                 f'MAE: {mae_15:.2f} kW\n'
                 f'MAPE: {mape_15:.2f} %\n'
                 f'Data Points: {n_15:,}')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.02, 0.98, textstr_15min, transform=ax1.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)

# --- Subplot 2: 1 hour ---
fechas_1h = [p.to_timestamp() for p in weekly_1h_real.index]
valores_real_1h = weekly_1h_real.values
valores_ppo_1h = weekly_1h_ppo.values

for year in all_years_1h:
    idx = [i for i, f in enumerate(fechas_1h) if f.year == year]
    if idx:
        c = year_colors.get(year, 'gray')
        fechas_year = [fechas_1h[i] for i in idx]
        ax2.plot(fechas_year, valores_real_1h[idx], color=c, linestyle='--', 
                 linewidth=1.5, marker='o', markersize=3, label=f'Actual {year}', alpha=0.8)
        ax2.plot(fechas_year, valores_ppo_1h[idx], color=c, linewidth=2, 
                 marker='s', markersize=3, label=f'Simulated {year}')

ax2.set_xlabel('Date', fontweight='bold')
ax2.set_ylabel('Average Power (kW)', fontweight='bold')
ax2.set_title(f'Interval: 1 hour ({year_range_1h})', fontweight='bold', fontsize=11)
ax2.legend(loc='upper right', ncol=2, frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Add dynamically calculated metrics
textstr_1h = (f'VALIDATION (ASHRAE G14)\n'
              f'CV(RMSE): {cvrmse_1h:.2f} %\n'
              f'NMBE: {nmbe_1h:.2f} %\n'
              f'RMSE: {rmse_1h:.2f} kW\n'
              f'MAE: {mae_1h:.2f} kW\n'
              f'MAPE: {mape_1h:.2f} %\n'
              f'Data Points: {n_1h:,}')
ax2.text(0.02, 0.98, textstr_1h, transform=ax2.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)

fig.suptitle('Model Validation - Complete Time Series - Tuesday', 
             fontweight='bold', fontsize=13, y=0.995)

plt.tight_layout()
output_file = os.path.join(base_path, 'series_temporales_martes_comparativa_EN_AllData.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Chart generated successfully with ALL data")
print(f"\n  15-min data: {year_range_15min} ({n_15:,} data points)")
print(f"  1-hour data: {year_range_1h} ({n_1h:,} data points)")
print(f"\nFile created:")
print(f"  {output_file}")
