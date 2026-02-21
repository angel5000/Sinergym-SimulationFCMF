import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from datetime import datetime
import matplotlib.dates as mdates
import os

print("GENERATING COMPARISON PLOTS FOR MONDAY...")

# --- 1. DATA LOADING ---
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

    print("Files loaded.")

except Exception as e:
    print(f"Error: {e}")
    exit()

# --- 2. MERGE ---
df_merged = pd.DataFrame()
df_merged['Act P (kW)'] = df_real['Act P (kW)']
df_merged['Power_kW_PPO'] = df_sim['Power_kW_PPO']
df_merged = df_merged.dropna()

def get_smooth_curve(x, y, n=300):
    try:
        spl = make_interp_spline(x, y, k=3)
        x_smooth = np.linspace(x.min(), x.max(), n)
        y_smooth = spl(x_smooth)
        return x_smooth, y_smooth
    except:
        return x, y

def add_metrics_box(ax):
    mae = np.mean(np.abs(df_merged['Act P (kW)'] - df_merged['Power_kW_PPO']))
    mape = np.mean(np.abs((df_merged['Act P (kW)'] - df_merged['Power_kW_PPO']) / df_merged['Act P (kW)'])) * 100
    textstr = '\n'.join((
        r'$\mathrm{MAE}=%.2f$ kW' % (mae, ),
        r'$\mathrm{MAPE}=%.2f$ %%' % (mape, )))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

# --- PLOT 1: AVERAGE PROFILE ---
df_merged['hour'] = df_merged.index.hour
hourly_mean = df_merged.groupby('hour').mean()

x = hourly_mean.index
xr, yr = get_smooth_curve(x, hourly_mean['Act P (kW)'].values)
xs, ys = get_smooth_curve(x, hourly_mean['Power_kW_PPO'].values)

plt.figure(figsize=(12, 6))
plt.plot(xr, yr, 'k--', linewidth=2, alpha=0.7, label='Real (Avg)')
plt.plot(xs, ys, 'g-', linewidth=2.5, label='PPO (Sim)')
plt.fill_between(xr, yr, ys, color='green', alpha=0.1, label='Difference')

add_metrics_box(plt.gca())

plt.title('Average Energy Profile - Monday (1H)', fontsize=16, fontweight='bold')
plt.xlabel('Hour of Day', fontsize=12, fontweight='bold')
plt.ylabel('Active Power (kW)', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.xticks(range(0, 24, 2))
plt.savefig('Monday_Average_Profile.png', dpi=300)
plt.close()

# --- PLOT 2: YEARLY COMPARISON ---
df_merged['year'] = df_merged.index.year
years = sorted(df_merged['year'].unique())

fig, axes = plt.subplots(1, len(years), figsize=(16, 6), sharey=True)
if len(years) == 1: axes = [axes]

for i, year in enumerate(years):
    df_year = df_merged[df_merged['year'] == year]
    hourly = df_year.groupby('hour').mean()
    
    x = hourly.index
    xr, yr = get_smooth_curve(x, hourly['Act P (kW)'].values)
    xs, ys = get_smooth_curve(x, hourly['Power_kW_PPO'].values)
    
    axes[i].plot(xr, yr, 'k--', label='Real', alpha=0.6)
    axes[i].plot(xs, ys, 'b-', label='PPO', linewidth=2)
    axes[i].set_title(f'Year {year}', fontweight='bold')
    axes[i].set_xlabel('Hour')
    axes[i].grid(True, alpha=0.3)
    if i == 0: 
        axes[i].set_ylabel('Power (kW)')
        axes[i].legend()

plt.suptitle('Yearly Validation - Monday', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('Monday_Yearly_Comparison.png', dpi=300)
plt.close()

# --- PLOT 3: TIME SERIES ---
daily_points = df_merged.groupby(df_merged.index.date).mean()
dates = pd.to_datetime(daily_points.index)

plt.figure(figsize=(18, 9))

x_days = np.arange(len(dates))
if len(x_days) > 10:
    xr, yr = get_smooth_curve(x_days, daily_points['Act P (kW)'].values, n=len(x_days)*3)
    xs, ys = get_smooth_curve(x_days, daily_points['Power_kW_PPO'].values, n=len(x_days)*3)
    date_range_smooth = pd.date_range(start=dates[0], end=dates[-1], periods=len(xr))
    
    plt.plot(date_range_smooth, yr, 'k--', linewidth=2, alpha=0.8, label='Real (Avg)')
    plt.plot(date_range_smooth, ys, 'b-', linewidth=2.5, alpha=0.9, label='PPO (Sim)')
else:
    plt.plot(dates, daily_points['Act P (kW)'], 'k--o', label='Real')
    plt.plot(dates, daily_points['Power_kW_PPO'], 'b-o', label='PPO')

add_metrics_box(plt.gca())

plt.title('Model Validation - Full Time Series - Monday (1H)', fontsize=16, fontweight='bold')
plt.ylabel('Daily Avg Power (kW)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.savefig('Monday_TimeSeries.png', dpi=300)
plt.close()
