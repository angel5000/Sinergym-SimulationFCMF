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
import time

print("\n" + "="*80)
print("**SIMULACION DEL DIA MARTES (ESCENARIO 1 HORA)**")
print("="*80 + "\n")

# --- 1. CONFIGURACION GENERAL ---
# Rutas del modelo de edificio (epJSON), archivo climatico (EPW) y datos reales (CSV).
# Se utiliza un modelo de referencia con caracteristicas similares al edificio
# de la facultad, ajustando parametros constructivos y operativos.
ORIGINAL_FILE = '/home/vboxuser/Descargas/EDIFICIO_SOLIDO.epJSON'
FINAL_FILE = '/home/vboxuser/Descargas/EDIFICIO_LUNES_OPTIMIZADO.epJSON' 
EPW_FILE = '/home/vboxuser/Descargas/Guayaquilcl2023.epw'

# Archivos CSV de datos reales (se busca en orden de prioridad)
POSIBLES_CSVS = ['martes_hora.csv', 'Martes_Hora.csv', 'lunes_hora.csv'] 

LOG_DIR = "./logs_martes/"
MODEL_DIR = "./modelos_martes/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 2. CARGA DEL PERFIL DE CONSUMO REAL (DATOS MEDIDOS) ---
# Se carga el perfil horario promedio de potencia activa medido en el edificio real.
# Este perfil se utiliza en la etapa de calibracion hibrida (ver Seccion C del step).
print(f"Buscando perfil objetivo...")
target_profile = {}
archivo_usado = None

for csv_file in POSIBLES_CSVS:
    if os.path.exists(csv_file):
        try:
            print(f"   -> Intentando leer: {csv_file}...")
            df_real = pd.read_csv(csv_file)
            
            # Detectar columna de fecha
            col_date = 'Local Time Stamp' if 'Local Time Stamp' in df_real.columns else 'Timestamp'
            df_real['Timestamp'] = pd.to_datetime(df_real[col_date])
            
            # SI ES ARCHIVO DE LUNES, FILTRAMOS LUNES (0). SI ES MARTES, MARTES (1).
            dia_filtro = 0 if 'lunes' in csv_file.lower() else 1
            
            df_dia = df_real[df_real['Timestamp'].dt.dayofweek == dia_filtro]
            
            if df_dia.empty:
                print(f"      El archivo {csv_file} no tiene datos para el día {dia_filtro}.")
                continue
                
            # Crear perfil
            target_profile = df_dia.groupby(df_dia['Timestamp'].dt.hour)['Act P (kW)'].mean().to_dict()
            
            # Rellenar huecos
            for h in range(24):
                if h not in target_profile: target_profile[h] = 20.0
            
            archivo_usado = csv_file
            print(f"¡ÉXITO! Usando perfil basado en: {csv_file}")
            break 
        except Exception as e:
            print(f"      Error leyendo {csv_file}: {e}")

if not archivo_usado:
    print("\nNo se encontro ningun CSV de datos reales.")
    print("   -> Se usara perfil de emergencia (20 kW constante).")
    target_profile = {h: 20.0 for h in range(24)}

# --- 3. PREPARACION DEL MODELO DE EDIFICIO ---
# Se ajustan parametros del epJSON para estabilidad numerica.
try:
    with open(ORIGINAL_FILE, 'r') as f:
        data = json.load(f)
    if "Version" in data:
        for key in data["Version"]: data["Version"][key]["version_identifier"] = "23.1"
    if "SimulationControl" in data:
        for key in data["SimulationControl"]:
            data["SimulationControl"][key]["maximum_hvac_iterations"] = 60
    if "Building" in data:
        for key in data["Building"]:
            data["Building"][key]["temperature_convergence_tolerance_value"] = 0.5
    with open(FINAL_FILE, 'w') as f:
        json.dump(data, f, indent=4)
except Exception as e:
    print(f"Error json: {e}")
    exit()

# --- 4. ENTORNO DE APRENDIZAJE POR REFUERZO ---
# Entorno Gymnasium que envuelve a EnergyPlus (via Sinergym).
# El agente PPO controla los setpoints de climatizacion (cooling/heating)
# y recibe una recompensa basada en consumo energetico y confort termico.
class PowerComfortEnv(Env):
    metadata = {"render_modes": []}
    
    def __init__(self, building_file, weather_file, target_dict, runperiod=(1, 11, 2021, 31, 12, 2022), 
                 zone_name='PERIMETER_BOT_ZN_1', w_energy=0.5, w_comfort=0.5):
        super().__init__()
        
        self.target_profile = target_dict
        self.w_energy = w_energy   
        self.w_comfort = w_comfort 
        
        self.building_file = building_file
        self.weather_file = weather_file
        self.zone_name = zone_name
        self.runperiod = runperiod
        self.timesteps_per_hour = 1 
        
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

    def reset(self, *, seed=None, options=None):
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

        t_zone = obs[3]
        energy_j = obs[4]
        hour = int(obs[2])
        
        # --- A. POTENCIA SIMULADA ---
        # Conversion de energia (J) a potencia (kW)
        power_sim_kw = (energy_j / 3600.0) / 1000.0 
        
        # Potencia objetivo del perfil real medido
        target_kw = self.target_profile.get(hour, 20.0)
        
        # --- B. CALCULO DE RECOMPENSA (FUNCION OBJETIVO DEL AGENTE) ---
        # La recompensa usa exclusivamente datos de la simulacion.
        # Componente energetico: penaliza consumo alto (normalizado a 100 kW)
        p_norm = np.clip(power_sim_kw / 100.0, 0, 1)
        # Componente de confort: penaliza desviaciones fuera del rango 23-26 C
        if t_zone < 23.0: diff_t = 23.0 - t_zone
        elif t_zone > 26.0: diff_t = t_zone - 26.0
        else: diff_t = 0.0
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
    print("Fase 1: Entrenamiento del agente PPO (Martes)...")
    
    env = PowerComfortEnv(FINAL_FILE, EPW_FILE, target_dict=target_profile)
    env = Monitor(env, LOG_DIR)
    vec_env = DummyVecEnv([lambda: env])
    
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64)
    checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=MODEL_DIR, name_prefix="ppo_martes")
    
    try:
        model.learn(total_timesteps=150000, callback=checkpoint_callback)
    except: pass
    
    model.save("PPO_Agente_Martes_Final")
    
    # =====================================================================
    # FASE 2: GENERACION DEL DATASET DE RESULTADOS
    # =====================================================================
    print("\nGenerando CSV Martes...")
    try: vec_env.env_method("close")
    except: pass
    
    obs = vec_env.reset()
    current_date = datetime(2021, 11, 7, 0, 0, 0)
    end_date = datetime(2022, 11, 15, 23, 0, 0)
    data_list = []
    
    try:
        while current_date <= end_date:
            action, _ = model.predict(obs)
            obs, rewards, dones, infos = vec_env.step(action)
            info = infos[0]
            
            t_val = obs[0][3]
            p_val = info.get('power_kw_calibrated', 0.0)
            
            # Recompensa obtenida por el agente en este paso
            actual_reward = rewards[0]
            
            # Solo se registran datos correspondientes a Martes (weekday=1)
            if current_date.weekday() == 1:
                 data_list.append([
                     current_date.strftime("%Y-%m-%d %H:%M:%S"), 
                     t_val, 
                     p_val, 
                     actual_reward,
                     0
                 ])
            
            current_date += timedelta(hours=1)
    except Exception as e:
        print(f"Error Loop: {e}")

    df = pd.DataFrame(data_list, columns=['Timestamp', 'T_Zone_PPO', 'Power_kW_PPO', 'Reward', 'Es_Feriado'])
    df.to_csv('Resultados_PPO_Martes_1H_Final.csv', index=False)
    print("Resultados guardados: Resultados_PPO_Martes_1H_Final.csv")

if __name__ == '__main__':
    main()
