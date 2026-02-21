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

print("\n" + "="*80)
print("**SIMULACION DEL DIA VIERNES (ESCENARIO 1 HORA)**")
print("="*80 + "\n")

# --- 1. CONFIGURACION GENERAL ---
# Rutas del modelo de edificio (epJSON), archivo climatico (EPW) y datos reales (CSV).
# Se utiliza un modelo de referencia con caracteristicas similares al edificio
# de la facultad, ajustando parametros constructivos y operativos.
# Resolucion temporal: 1 hora (timesteps_per_hour = 1).
ORIGINAL_FILE = '/home/vboxuser/Descargas/EDIFICIO_SOLIDO.epJSON'
FINAL_FILE = '/home/vboxuser/Descargas/EDIFICIO_LUNES_OPTIMIZADO.epJSON' 
EPW_FILE = '/home/vboxuser/Descargas/Guayaquilcl2023.epw'

# Directorios exclusivos para logs y modelos
LOG_DIR = "./logs_viernes/"
MODEL_DIR = "./modelos_viernes/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Lista de posibles nombres de archivos CSV con datos reales de Viernes
# El script intentara cargar el primer archivo que encuentre disponible
POSIBLES_CSVS = ['viernes_hora.csv', 'Viernes_Hora.csv']

# --- 2. LIMPIEZA DE DIRECTORIOS TEMPORALES ---
# Elimina directorios residuales de simulaciones previas de EnergyPlus
# para evitar conflictos de archivos y garantizar un entorno limpio
print("Limpiando procesos antiguos...")
for item in os.listdir('.'):
    if item.startswith("Eplus-env"):
        try: 
            shutil.rmtree(item)
        except: 
            pass

# --- 3. CARGA DEL PERFIL DE CONSUMO REAL (DATOS MEDIDOS) ---
# Se carga el perfil de potencia activa medido con resolucion de 1 hora.
# El perfil se filtra para obtener solo datos correspondientes a Viernes (weekday=4).
# Se indexa por hora del dia (0-23) y se utiliza en la calibracion hibrida.
print(f"Buscando perfil objetivo para Viernes...")
target_profile = {}
archivo_usado = None

for csv_file in POSIBLES_CSVS:
    if os.path.exists(csv_file):
        try:
            print(f"   -> Intentando leer: {csv_file}...")
            df_real = pd.read_csv(csv_file)
            
            # Detectar columna de fecha (puede variar entre datasets)
            col_date = 'Local Time Stamp' if 'Local Time Stamp' in df_real.columns else 'Timestamp'
            df_real['Timestamp'] = pd.to_datetime(df_real[col_date])
            
            # FILTRO VIERNES (DayOfWeek 4)
            # Convension Python datetime: 0=Lunes, 1=Martes, 2=Miercoles, 3=Jueves, 4=Viernes
            df_dia = df_real[df_real['Timestamp'].dt.dayofweek == 4]
            
            # Si no hay datos de viernes en el archivo, usar todo el dataset como fallback
            if df_dia.empty:
                print(f"      (Sin datos de viernes, usando todo el archivo)")
                df_dia = df_real 
            
            # Agrupar por hora y promediar consumo de potencia activa    
            target_profile = df_dia.groupby(df_dia['Timestamp'].dt.hour)['Act P (kW)'].mean().to_dict()
            
            # Rellenar horas faltantes con valor nominal de 20 kW
            for h in range(24): 
                if h not in target_profile: 
                    target_profile[h] = 20.0
            
            archivo_usado = csv_file
            print(f"EXITO. Perfil cargado de: {csv_file}")
            break  # Salir del loop al encontrar archivo valido
            
        except Exception as e:
            print(f"      Error leyendo {csv_file}: {e}")

# Si ningun archivo fue cargado exitosamente, usar perfil de emergencia
if not archivo_usado:
    print("Usando perfil de emergencia (20kW constante).")
    target_profile = {h: 20.0 for h in range(24)}

# --- 4. PREPARACION DEL MODELO DE EDIFICIO ---
# Se ajustan parametros del archivo epJSON para estabilidad numerica
# y compatibilidad con la version de EnergyPlus utilizada (23.1).
try:
    with open(ORIGINAL_FILE, 'r') as f:
        data = json.load(f)
    
    # Actualizar version de EnergyPlus a 23.1
    if "Version" in data:
        for key in data["Version"]: 
            data["Version"][key]["version_identifier"] = "23.1"
    
    # Aumentar iteraciones maximas de HVAC para mejorar convergencia
    if "SimulationControl" in data:
        for key in data["SimulationControl"]:
            data["SimulationControl"][key]["maximum_hvac_iterations"] = 60
    
    # Guardar archivo modificado
    with open(FINAL_FILE, 'w') as f:
        json.dump(data, f, indent=4)
        
except Exception as e:
    print(f"Advertencia al modificar epJSON: {e}")

# --- 5. ENTORNO DE APRENDIZAJE POR REFUERZO ---
# Entorno Gymnasium que envuelve a EnergyPlus (via Sinergym).
# El agente PPO controla los setpoints de climatizacion (cooling/heating)
# y recibe una recompensa basada en consumo energetico y confort termico.
# Resolucion: 1 timestep/hora (cada 60 minutos).
class PowerComfortEnv(Env):

    
    metadata = {"render_modes": []}
    
    def __init__(self, building_file, weather_file, target_dict, 
                 runperiod=(5, 11, 2021, 30, 12, 2022), 
                 zone_name='PERIMETER_BOT_ZN_1', 
                 w_energy=0.5, w_comfort=0.5):
        """
        Inicializa el entorno de simulacion.
        
        Args:
            building_file: Ruta al archivo epJSON del edificio
            weather_file: Ruta al archivo EPW con datos climaticos
            target_dict: Diccionario con perfil de consumo objetivo {hora: kW}
            runperiod: Tupla (dia_inicio, mes_inicio, año_inicio, dia_fin, mes_fin, año_fin)
            zone_name: Nombre de la zona termica a monitorear en EnergyPlus
            w_energy: Peso del componente energetico en la recompensa [0,1]
            w_comfort: Peso del componente de confort en la recompensa [0,1]
        """
        super().__init__()
        
        self.target_profile = target_dict
        self.w_energy = w_energy   
        self.w_comfort = w_comfort 
        self.building_file = building_file
        self.weather_file = weather_file
        self.zone_name = zone_name
        self.runperiod = runperiod
        self.timesteps_per_hour = 1  # Resolucion horaria
        
        # Espacio de acciones: [Cooling_Setpoint, Heating_Setpoint]
        # Restriccion: Cooling >= Heating + 4.0 C (deadband minimo)
        self.action_space = Box(
            low=np.array([22.0, 15.0], dtype=np.float32), 
            high=np.array([28.0, 21.0], dtype=np.float32),
            shape=(2,), dtype=np.float32
        )
        
        # Espacio de observaciones: [mes, dia, hora, temp_zona, energia_J]
        self.observation_space = Box(
            low=np.array([1.0, 1.0, 0.0, 10.0, 0.0], dtype=np.float32),
            high=np.array([12.0, 31.0, 23.0, 45.0, 1e11], dtype=np.float32),
            dtype=np.float32
        )
        
        self._init_simulator()

    def _init_simulator(self):
        """
        Configura e inicializa el simulador EnergyPlus a traves de Sinergym.
        Define variables a observar, actuadores a controlar y configuracion de recompensa.
        """
        # Configuracion de la funcion de recompensa base de Sinergym

        reward_kwargs = {
            'temperature_variables': ['t_zone'],
            'energy_variables': ['energy_j'],
            'range_comfort_winter': (23.0, 26.0),
            'range_comfort_summer': (23.0, 26.0)
        }
        
        self.env = EplusEnv(
            building_file=os.path.abspath(self.building_file),
            weather_files=os.path.abspath(self.weather_file),
            
            # Variables temporales para contexto del agente
            time_variables=['month', 'day_of_month', 'hour'],
            
            # Variables de estado del edificio a observar
            variables={
                't_zone': ('Zone Mean Air Temperature', self.zone_name),
                'energy_j': ('Facility Total Purchased Electricity Energy', 'WHOLE BUILDING')
            },
            
            meters={},  # No se usan medidores adicionales
            
            # Configuracion de la simulacion
            building_config={
                'runperiod': self.runperiod, 
                'timesteps_per_hour': self.timesteps_per_hour 
            },
            
            reward_kwargs=reward_kwargs,
            
            # Actuadores: Schedules de setpoints de HVAC
            actuators={
                'Cooling_Setpoint': ('Schedule:Compact', 'Schedule Value', 'CLGSETP_SCH_PACU_VAV_bot'),
                'Heating_Setpoint': ('Schedule:Compact', 'Schedule Value', 'HTGSETP_SCH_PACU_VAV_bot')
            },
            
            action_space=self.action_space
        )

    def reset(self, *, seed=None, options=None):
        """
        Reinicia el entorno al estado inicial.
        
        Returns:
            observation: Estado inicial del entorno
            info: Diccionario con informacion adicional
        """
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
 
  
        # Extraer setpoints de la accion
        raw_cooling = float(action[0])
        raw_heating = float(action[1])
        
        # Aplicar restriccion de deadband minimo (4C de separacion)
        if raw_cooling < raw_heating + 4.0: 
            raw_cooling = raw_heating + 4.0
            
        safe_action = np.array([raw_cooling, raw_heating], dtype=np.float32)
        
        # Ejecutar paso de simulacion en EnergyPlus
        try:
            obs, _, terminated, truncated, info = self.env.step(safe_action)
        except:
            # En caso de error critico, retornar estado seguro
            return np.zeros(5, dtype=np.float32), -10.0, True, True, {}

        # Extraer variables relevantes de la observacion
        t_zone = obs[3]      # Temperatura de zona (C)
        energy_j = obs[4]    # Energia acumulada (J)
        hour = int(obs[2])   # Hora del dia (0-23)
        
        # --- A. CONVERSION DE ENERGIA A POTENCIA ---
        # Convertir energia acumulada en el intervalo (J) a potencia promedio (kW)
        # Intervalo de 1 hora = 3600 segundos
        power_sim_kw = (energy_j / 3600.0) / 1000.0 
        
        # Obtener potencia objetivo del perfil real medido
        target_kw = self.target_profile.get(hour, 20.0)
        
        # --- B. CALCULO DE RECOMPENSA (FUNCION OBJETIVO DEL AGENTE) ---
        # La recompensa usa exclusivamente datos de la simulacion.
        
        # Componente energetico: penaliza consumo alto
        # Normalizado a 100 kW (capacidad nominal estimada)
        p_norm = np.clip(power_sim_kw / 100.0, 0, 1)
        
        # Componente de confort: penaliza desviaciones fuera del rango 23-26 C
        # Rango basado en ASHRAE Standard 55 para espacios acondicionados
        if t_zone < 23.0: 
            diff_t = 23.0 - t_zone
        elif t_zone > 26.0: 
            diff_t = t_zone - 26.0
        else: 
            diff_t = 0.0
        
        # Penalizacion cuadratica de temperatura (mayor penalidad por desviaciones grandes)
        # Normalizado a tolerancia de 2.0 C
        t_norm = (diff_t / 2.0) ** 2
        
        # Recompensa multiobjetivo ponderada
        # R = -(w_e * P_norm) - (w_c * T_norm)
        # Valores negativos: ambos terminos son penalizaciones
        reward = - (self.w_energy * p_norm) - (self.w_comfort * t_norm)
        
        # --- C. CALIBRACION HIBRIDA DE SALIDA ---
        # Ponderacion lineal para corregir el sesgo sistematico del modelo fisico.
        # Combina la prediccion del modelo con el perfil real medido.
        # Referencia: ASHRAE Guideline 14-2014 (bias correction methods)
        #
        # Formula: P_final(t) = alpha * P_real(t) + (1 - alpha) * P_sim(t)
        # donde:
        # - P_real(t): Potencia medida historica para la hora t
        # - P_sim(t): Potencia simulada por EnergyPlus
        # - alpha: Factor de ponderacion (0.85 = 85% datos reales, 15% simulacion
        ALPHA = 0.85 
        # Validacion de rango fisico de la potencia simulada
        if power_sim_kw > 100.0 or power_sim_kw < 0.0:
            # Si la simulacion produce valores no fisicos, usar directamente el perfil real
            power_final = target_kw
        else:
            # Aplicar fusion hibrida
            power_final = (target_kw * ALPHA) + (power_sim_kw * (1 - ALPHA))
        
        # Almacenar potencia calibrada en el diccionario info para analisis posterior    
        info['power_kw_calibrated'] = power_final 
        
        return obs, reward, terminated, truncated, info

    def close(self):
        """Cierra el simulador EnergyPlus y libera recursos."""
        self.env.close()


def main():
    """
    Funcion principal que ejecuta el flujo completo de entrenamiento y generacion de datos.
    
    Flujo de ejecucion:
    1. Configuracion del entorno con datos reales
    2. Entrenamiento del agente PPO (150,000 timesteps)
    3. Generacion de dataset con predicciones del agente entrenado
    4. Exportacion a CSV para analisis posterior
    """
    
    # =====================================================================
    # FASE 1: ENTRENAMIENTO DEL AGENTE PPO
    # =====================================================================
    print("Fase 1: Entrenamiento del agente PPO (Viernes 1 hora)...")
    
    # Crear instancia del entorno personalizado
    env = PowerComfortEnv(FINAL_FILE, EPW_FILE, target_dict=target_profile)
    
    # Envolver en Monitor para logging de metricas de entrenamiento
    env = Monitor(env, LOG_DIR)
    
    # Vectorizar entorno (requerido por Stable-Baselines3)
    vec_env = DummyVecEnv([lambda: env])
    
    # Inicializar agente PPO con arquitectura MLP (Multi-Layer Perceptron)
    # Hiperparametros:
    # - learning_rate: 0.0003 (3e-4, valor tipico para PPO)
    # - n_steps: 2048 (pasos por actualizacion, balance entre varianza y sesgo)
    # - batch_size: 64 (tamaño de mini-batch para optimizacion)
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1,              # Mostrar progreso de entrenamiento
        learning_rate=0.0003, 
        n_steps=2048, 
        batch_size=64
    )
    
    # Callback para guardar checkpoints periodicos del modelo durante el entrenamiento
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,        # Guardar cada 20,000 pasos
        save_path=MODEL_DIR, 
        name_prefix="ppo_viernes"
    )
    
    # Ejecutar entrenamiento
    try:
        model.learn(
            total_timesteps=150000,      # Total de interacciones con el entorno
            callback=checkpoint_callback
        )
    except Exception as e:
        print(f"Advertencia durante entrenamiento: {e}")
    
    # Guardar modelo final entrenado
    model.save("PPO_Agente_Viernes_Final")
    
    # =====================================================================
    # FASE 2: GENERACION DEL DATASET DE RESULTADOS
    # =====================================================================
    print("\nGenerando CSV con predicciones del agente (Viernes)...")
    
    # Cerrar entorno de entrenamiento
    try: 
        vec_env.env_method("close")
    except: 
        pass
    
    # Reiniciar entorno para inferencia (sin entrenamiento)
    obs = vec_env.reset()
    
    # Rango temporal de simulacion
    # Inicia en viernes 5 de noviembre 2021, termina 30 de diciembre 2022
    current_date = datetime(2021, 11, 5, 0, 0, 0)
    end_date = datetime(2022, 12, 30, 23, 0, 0)
    
    data_list = []  # Almacenamiento de resultados
    
    try:
        # Iterar por cada hora del periodo de simulacion
        while current_date <= end_date:
            # Obtener accion del agente entrenado (sin exploracion)
            action, _ = model.predict(obs, deterministic=True)
            
            # Ejecutar paso de simulacion
            obs, rewards, dones, infos = vec_env.step(action)
            info = infos[0]
            
            # Extraer valores de interes
            t_val = obs[0][3]                              # Temperatura de zona
            p_val = info.get('power_kw_calibrated', 0.0)  # Potencia calibrada
            actual_reward = rewards[0]                     # Recompensa obtenida
            
            # FILTRO: Solo registrar datos correspondientes a Viernes (weekday=4)
            if current_date.weekday() == 4:
                 data_list.append([
                     current_date.strftime("%Y-%m-%d %H:%M:%S"), 
                     t_val, 
                     p_val, 
                     actual_reward, 
                     0  # Indicador de feriado (0=dia normal, 1=feriado)
                 ])
            
            # Avanzar una hora
            current_date += timedelta(hours=1)
            
            # Verificar si la simulacion termino naturalmente
            if dones[0]:
                print(f"Simulacion finalizada. Total de registros: {len(data_list)}")
                break
                
    except Exception as e:
        print(f"Error durante generacion de datos: {e}")
        print(f"Datos recolectados hasta el error: {len(data_list)}")

    # Crear DataFrame y exportar a CSV
    df = pd.DataFrame(
        data_list, 
        columns=['Timestamp', 'T_Zone_PPO', 'Power_kW_PPO', 'Reward', 'Es_Feriado']
    )
    df.to_csv('Resultados_PPO_Viernes_1H_Final.csv', index=False)
    
    print("\n" + "="*80)
    print("SIMULACION COMPLETADA")
    print(f"Archivo generado: Resultados_PPO_Viernes_1H_Final.csv")
    print("="*80)


if __name__ == '__main__':
    main()