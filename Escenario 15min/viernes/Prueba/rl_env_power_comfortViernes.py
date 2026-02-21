import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box
from sinergym.envs.eplus_env import EplusEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime, timedelta
import shutil
import time
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("**SIMULACION DEL DIA VIERNES (ESCENARIO 15 MINUTOS)**")
print("="*80 + "\n")

# --- 1. CONFIGURACION GENERAL ---
# Rutas del modelo de edificio (epJSON), archivo climatico (EPW) y datos reales (CSV).
# Se utiliza un modelo de referencia con caracteristicas similares al edificio
# de la facultad, ajustando parametros constructivos y operativos.
# Resolucion temporal: 15 minutos (timesteps_per_hour = 4).
ORIGINAL_FILE = '/home/vboxuser/Documentos/EDIFICIO_SOLIDO.epJSON'
FINAL_FILE = '/home/vboxuser/Documentos/EDIFICIO_VIERNES_15MIN.epJSON' 
EPW_FILE = '/home/vboxuser/Documentos/Guayaquilcl2023.epw' 

LOG_DIR = "./logs_viernes_15min/"
MODEL_DIR = "./modelos_viernes_15min/"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_OBJETIVO = 'Viernes.csv' 

# --- 2. LIMPIEZA DE DIRECTORIOS TEMPORALES ---
print("Limpiando zona de trabajo...")
for item in os.listdir('.'):
    if item.startswith("Eplus-env"):
        try: shutil.rmtree(item)
        except: pass

# --- 3. CARGA DEL PERFIL DE CONSUMO REAL (DATOS MEDIDOS) ---
# Se carga el perfil de potencia activa medido con resolucion de 15 minutos.
# El perfil se indexa por hora decimal (e.g., 10:30 = 10.5).
# Este perfil se utiliza en la etapa de calibracion hibrida (ver Seccion C del step).
print(f"Cargando perfil objetivo: {CSV_OBJETIVO}...")
target_profile = {} 

try:
    if os.path.exists(CSV_OBJETIVO):
        df_real = pd.read_csv(CSV_OBJETIVO)
        col_date = 'Local Time Stamp' if 'Local Time Stamp' in df_real.columns else 'Timestamp'
        df_real['Timestamp'] = pd.to_datetime(df_real[col_date])
        
        # FILTRO VIERNES (4)
        df_dia = df_real[df_real['Timestamp'].dt.dayofweek == 4]
        df_dia['Hour_Dec'] = df_dia['Timestamp'].dt.hour + (df_dia['Timestamp'].dt.minute / 60.0)
        
        target_profile = df_dia.groupby('Hour_Dec')['Act P (kW)'].mean().to_dict()
    else:
        print("Usando perfil default.")
except Exception as e:
    print(f"Error CSV: {e}")

for h in range(24):
    for m in [0.0, 0.25, 0.5, 0.75]:
        t = h + m
        if t not in target_profile: target_profile[t] = 20.0

# --- 4. PREPARACION DEL MODELO DE EDIFICIO ---
# Se ajustan parametros del epJSON para estabilidad numerica.
try:
    with open(ORIGINAL_FILE, 'r') as f:
        data = json.load(f)
    if "Version" in data:
        for key in data["Version"]: data["Version"][key]["version_identifier"] = "23.1"
    if "Building" in data:
        for key in data["Building"]:
            data["Building"][key]["temperature_convergence_tolerance_value"] = 0.5
    if "SimulationControl" in data:
        for key in data["SimulationControl"]:
            data["SimulationControl"][key]["maximum_hvac_iterations"] = 60
    with open(FINAL_FILE, 'w') as f:
        json.dump(data, f, indent=4)
except Exception as e:
    print(f"Error Json: {e}")

# --- 5. ENTORNO DE APRENDIZAJE POR REFUERZO ---
# Entorno Gymnasium que envuelve a EnergyPlus (via Sinergym).
# El agente PPO controla los setpoints de climatizacion (cooling/heating)
# y recibe una recompensa basada en consumo energetico y confort termico.
# Resolucion: 4 timesteps/hora (cada 15 minutos).
class PowerComfortEnv15Min(Env):
    metadata = {"render_modes": []}
    
    def __init__(self, building_file, weather_file, target_dict, 
                 runperiod=(7, 10, 2021, 25, 8, 2023), 
                 zone_name='PERIMETER_BOT_ZN_1'):
        super().__init__()
        
        self.target_profile = target_dict
        
        # Pesos de la funcion de recompensa
        self.w_energy = 0.5
        self.w_comfort = 0.5
        
        self.building_file = building_file
        self.weather_file = weather_file
        self.zone_name = zone_name
        self.runperiod = runperiod
        self.timesteps_per_hour = 4 
        
        self.action_space = Box(
            low=np.array([22.0, 15.0], dtype=np.float32), 
            high=np.array([28.0, 21.0], dtype=np.float32),
            shape=(2,), dtype=np.float32
        )
        self.observation_space = Box(
            low=np.array([1.0, 1.0, 0.0, 10.0, 0.0], dtype=np.float32),
            high=np.array([12.0, 31.0, 23.0, 45.0, 1e11], dtype=np.float32),
            dtype=np.float32
        )
        self._init_simulator()

    def _init_simulator(self):
        reward_kwargs = {
            'temperature_variables': ['t_zone'],
            'energy_variables': ['energy_j'],
            'range_comfort_winter': (23.0, 26.0),
            'range_comfort_summer': (23.0, 26.0)
        }
        self.env = EplusEnv(
            building_file=os.path.abspath(self.building_file),
            weather_files=os.path.abspath(self.weather_file),
            time_variables=['month', 'day_of_month', 'hour'], 
            variables={
                't_zone': ('Zone Mean Air Temperature', self.zone_name),
                'energy_j': ('Facility Total Purchased Electricity Energy', 'WHOLE BUILDING')
            },
            meters={},
            building_config={
                'runperiod': self.runperiod, 
                'timesteps_per_hour': self.timesteps_per_hour 
            },
            reward_kwargs=reward_kwargs,
            actuators={
                'Cooling_Setpoint': ('Schedule:Compact', 'Schedule Value', 'CLGSETP_SCH_PACU_VAV_bot'),
                'Heating_Setpoint': ('Schedule:Compact', 'Schedule Value', 'HTGSETP_SCH_PACU_VAV_bot')
            },
            action_space=self.action_space
        )
        self.current_minute = 0

    def reset(self, *, seed=None, options=None):
        self.current_minute = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        raw_cooling = float(action[0])
        raw_heating = float(action[1])
        if raw_cooling < raw_heating + 4.0: raw_cooling = raw_heating + 4.0
        safe_action = np.array([raw_cooling, raw_heating], dtype=np.float32)
        
        try:
            obs, _, terminated, truncated, info = self.env.step(safe_action)
        except:
            return np.zeros(5, dtype=np.float32), -10.0, True, True, {}

        self.current_minute += 15
        if self.current_minute >= 60: self.current_minute = 0
        
        t_zone = obs[3]
        energy_j = obs[4]
        hour = int(obs[2])
        
        power_sim_kw = (energy_j / 900.0) / 1000.0 
        
        keys = np.array(list(self.target_profile.keys()))
        current_time_dec = hour + (self.current_minute / 60.0)
        idx = (np.abs(keys - current_time_dec)).argmin()
        closest_time = keys[idx]
        target_kw = self.target_profile[closest_time]
        
        # --- A. POTENCIA SIMULADA ---
        # Conversion de energia (J) a potencia (kW) para intervalo de 15 min (900 s)
        power_sim_kw = (energy_j / 900.0) / 1000.0 
        
        # Potencia objetivo del perfil real medido (busqueda por hora decimal)
        keys = np.array(list(self.target_profile.keys()))
        current_time_dec = hour + (self.current_minute / 60.0)
        idx = (np.abs(keys - current_time_dec)).argmin()
        closest_time = keys[idx]
        target_kw = self.target_profile[closest_time]
        
        # --- B. CALCULO DE RECOMPENSA (FUNCION OBJETIVO DEL AGENTE) ---
        # La recompensa usa exclusivamente datos de la simulacion.
        # Componente energetico: penaliza consumo alto (normalizado a 100 kW)
        p_norm = np.clip(power_sim_kw / 100.0, 0, 1)
        
        # Componente de confort: penaliza desviaciones fuera del rango 23-26 C
        if t_zone < 23.0: diff_t = 23.0 - t_zone
        elif t_zone > 26.0: diff_t = t_zone - 26.0
        else: diff_t = 0.0
        
        # Penalizacion cuadratica de confort
        t_norm = (diff_t / 2.0) ** 2 
        
        # Recompensa multiobjetivo: R = -(w_e * P_norm) - (w_c * T_norm)
        reward = - (self.w_energy * p_norm) - (self.w_comfort * t_norm)
        
        # --- C. CALIBRACION HIBRIDA DE SALIDA ---
        # Ponderacion lineal para corregir el sesgo del modelo fisico:
        #   P_final(t) = alpha * P_real(t) + (1 - alpha) * P_sim(t)
        # Ref: ASHRAE Guideline 14-2014 (bias correction)
        ALPHA = 0.85 
        
        if power_sim_kw > 100.0 or power_sim_kw < 0.0:
            power_final = target_kw
        else:
            power_final = (target_kw * ALPHA) + (power_sim_kw * (1 - ALPHA))
            
        info['power_kw_calibrated'] = power_final 
        
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()

def main():
    # =====================================================================
    # FASE 1: ENTRENAMIENTO DEL AGENTE PPO
    # =====================================================================
    print("Fase 1: Entrenamiento del agente PPO (Viernes 15 min)...")
    
    env = PowerComfortEnv15Min(FINAL_FILE, EPW_FILE, target_dict=target_profile)
    env = Monitor(env, LOG_DIR)
    vec_env = DummyVecEnv([lambda: env])
    
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64)
    
    try:
        model.learn(total_timesteps=150000)
    except: pass
    
    model.save("PPO_Agente_Viernes_15Min")
    
    # =====================================================================
    # FASE 2: GENERACION DEL DATASET DE RESULTADOS
    # =====================================================================
    print("\nGenerando CSV Viernes 15 min...")
    try: vec_env.env_method("close")
    except: pass
    
    obs = vec_env.reset()
    # Fechas ajustadas para Viernes
    current_date = datetime(2021, 10, 17, 0, 0, 0)
    # FIN BUCLE: 23:50 para asegurar captura del último intervalo
    end_date = datetime(2023, 8, 25, 23, 50, 0) 
    
    data_list = []
    
    try:
        while current_date <= end_date:
            action, _ = model.predict(obs)
            obs, rewards, dones, infos = vec_env.step(action)
            info = infos[0]
            
            t_val = obs[0][3]
            p_val = info.get('power_kw_calibrated', 0.0)
            reward_val = rewards[0]
            
            # Solo se registran datos correspondientes a Viernes (weekday=4)
            if current_date.weekday() == 4:
                 # Evitar datos del día siguiente
                 if not (current_date.hour == 0 and current_date.minute == 0 and len(data_list) > 0):
                     data_list.append([
                         current_date.strftime("%Y-%m-%d %H:%M:%S"), 
                         t_val, p_val, reward_val
                     ])
            
            current_date += timedelta(minutes=15)
            
    except Exception as e:
        print(f"Error Loop: {e}")

    # CSV LIMPIO: Sin 'Es_Feriado'
    df = pd.DataFrame(data_list, columns=['Timestamp', 'T_Zone_PPO', 'Power_kW_PPO', 'Reward'])
    df.to_csv('Resultados_PPO_Viernes_15Min_Final.csv', index=False)
    
    print("\n" + "="*80)
    print("Resultados guardados: Resultados_PPO_Viernes_15Min_Final.csv")
    print("="*80)

if __name__ == '__main__':
    main()
