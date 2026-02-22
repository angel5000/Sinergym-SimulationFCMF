import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend (avoids Tcl/Tk error)
import matplotlib.pyplot as plt
from datetime import datetime

# Configure style for academic graphs
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# ============================================================================
# LOAD DATA (ONLY PPO RESULTS FOR COMPARISON)
# ============================================================================

# 15-minute data
df_ppo_15min = pd.read_csv('./Resultados_PPO_Lunes_15Min_Final.csv')
df_ppo_15min['PPO_Potencia_kW'] = df_ppo_15min['Power_kW_PPO']
df_ppo_15min['Fecha'] = pd.to_datetime(df_ppo_15min['Timestamp'])

# 1-hour data
df_ppo_1h = pd.read_csv('./Resultados_PPO_Lunes_1H_Final.csv')
df_ppo_1h['PPO_Potencia_kW'] = df_ppo_1h['Power_kW_PPO']
df_ppo_1h['Fecha'] = pd.to_datetime(df_ppo_1h['Timestamp'])

# ============================================================================
# CHART: TEMPORAL RESOLUTION COMPARISON (SINGLE-DAY EXAMPLE)
# ============================================================================

# Select a specific day for comparison (a Monday)
fecha_ejemplo = '2022-01-17'  # A Monday
df_15min_dia = df_ppo_15min[df_ppo_15min['Fecha'].dt.date == pd.to_datetime(fecha_ejemplo).date()].copy()
df_1h_dia = df_ppo_1h[df_ppo_1h['Fecha'].dt.date == pd.to_datetime(fecha_ejemplo).date()].copy()

# Extract hour of the day
df_15min_dia['Hora'] = df_15min_dia['Fecha'].dt.hour + df_15min_dia['Fecha'].dt.minute/60
df_1h_dia['Hora'] = df_1h_dia['Fecha'].dt.hour

# Create chart
fig, ax = plt.subplots(figsize=(12, 5))

# Plot data
ax.plot(df_15min_dia['Hora'], df_15min_dia['PPO_Potencia_kW'], 
        color='lightblue', linewidth=1.5, label='Scenario A: 15 Minutes', alpha=0.7)
ax.plot(df_1h_dia['Hora'], df_1h_dia['PPO_Potencia_kW'], 
        color='orangered', linewidth=2.5, marker='o', markersize=4,
        label='Scenario B: 1 Hour')

ax.set_xlabel('Hour of the Day', fontweight='bold')
ax.set_ylabel('Active Power (kW)', fontweight='bold')
ax.set_title(f'Temporal Resolution Comparison: 15 min vs 1 Hour\n(Day Example: {fecha_ejemplo})', 
             fontweight='bold', pad=15)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 24)
ax.set_xticks(range(0, 25, 2))

plt.tight_layout()
output_name = 'comparativa_resolucion_temporal_ejemplo_EN.png'
plt.savefig(output_name, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Chart generated successfully: {output_name}")
