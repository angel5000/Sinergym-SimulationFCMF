#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path
from datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from eppy.modeleditor import IDF
import sys

# ---------------------------
# 1. CONFIGURACION GENERAL
# ---------------------------
IDF_ORIGINAL = Path("ASHRAE901_OfficeMedium_STD2019_Miami.idf")
IDF_TRABAJO = Path("EDIFICIO_AHORRO_V23.idf") 
EPW_FILE = Path("Guayaquilcl2023.epw")
REAL_DATA_FILE = Path("real_power_kw_15min.csv") 

# Ruta del motor EnergyPlus v23.1
EPLUS_DIR = Path("/usr/local/EnergyPlus-23-1-0")
EPLUS_EXE = EPLUS_DIR / "energyplus"
IDD_FILE = EPLUS_DIR / "Energy+.idd"

OUTPUT_ROOT = Path("./salidas_ajuste_ahorro")

# Rango de fechas para la simulacion.
# Se usa 2023 para alinear el calendario con los datos reales del CSV (Enero 2023).
START_YEAR = 2023 
START_MONTH = 1
START_DAY = 9 # Lunes 9 de Enero 2023
N_DAYS = 7

METER_NAME = "Electricity:Facility"

# ---------------------------
# 2. LIMPIEZA DE ARCHIVOS PREVIOS
# ---------------------------
def clean_previous_failures():
    """Restaura el IDF de trabajo a partir del original para partir de un estado limpio."""
    print("LIMPIANDO ARCHIVOS...")
    if IDF_ORIGINAL.exists():
        shutil.copy(IDF_ORIGINAL, IDF_TRABAJO)

# ---------------------------
# 3. TRANSICION DEL IDF A V23
# ---------------------------
def update_to_v23():
    updater_dir = EPLUS_DIR / "PreProcess" / "IDFVersionUpdater"
    transitions = ["Transition-V22-1-0-to-V22-2-0", "Transition-V22-2-0-to-V23-1-0"]
    with open(IDF_TRABAJO, 'r', errors='ignore') as f:
        if "Version,23.1" in f.read(): return
    
    work_dir = Path("./temp_fix")
    work_dir.mkdir(exist_ok=True)
    temp_file = work_dir / IDF_TRABAJO.name
    shutil.copy(IDF_TRABAJO, temp_file)
    for trans in transitions:
        exe = shutil.which(trans, path=str(updater_dir))
        if not exe and (updater_dir / trans).exists(): exe = str(updater_dir / trans)
        if exe: subprocess.run([exe, str(temp_file)], cwd=work_dir, capture_output=True)
    if temp_file.exists(): shutil.copy(temp_file, IDF_TRABAJO)

# ---------------------------
# 4. CALIBRACION AGRESIVA (Modo Ahorro)
# ---------------------------
def set_param_safe(obj, partial_name, value):
    for field in obj.fieldnames:
        if partial_name.lower() in field.lower():
            if "high" in partial_name.lower() and "low" in field.lower(): continue
            if "low" in partial_name.lower() and "high" in field.lower(): continue
            obj[field] = value
            return True
    return False

def get_param_safe(obj, partial_name):
    for field in obj.fieldnames:
        if partial_name.lower() in field.lower():
            if "high" in partial_name.lower() and "low" in field.lower(): continue
            if "low" in partial_name.lower() and "high" in field.lower(): continue
            return obj[field]
    return None

def apply_aggressive_reduction(idf):
    """Aplica reducciones agresivas de consumo energetico al modelo IDF."""
    print("APLICANDO REDUCCION AGRESIVA DE CONSUMO...")

    # 1. CERRAR EL AIRE EXTERIOR (Recirculacion)
    # Critico en Guayaquil: el aire exterior caliente y humedo dispara la carga latente.
    print("   -> Cerrando tomas de aire exterior (Modo Recirculacion)...")
    if idf.idfobjects['DESIGNSPECIFICATION:OUTDOORAIR']:
        for oa in idf.idfobjects['DESIGNSPECIFICATION:OUTDOORAIR']:
            # Ventilacion reducida al 15% del valor normativo.
            # Suficiente para evitar errores, pero reduce considerablemente la carga latente.
            val_person = get_param_safe(oa, "Outdoor Air Flow per Person")
            if val_person: set_param_safe(oa, "Outdoor Air Flow per Person", float(val_person) * 0.15)
            
            val_area = get_param_safe(oa, "Outdoor Air Flow per Zone Floor Area")
            if val_area: set_param_safe(oa, "Outdoor Air Flow per Zone Floor Area", float(val_area) * 0.15)

    # 2. SUBIR TERMOSTATO DE ENFRIAMIENTO A 25 C
    print("   -> Subiendo termostatos a 25 C...")
    if idf.idfobjects['SCHEDULE:COMPACT']:
        for sched in idf.idfobjects['SCHEDULE:COMPACT']:
            if "clg" in sched.Name.lower() or "cooling" in sched.Name.lower():
                # Busca valores tipicos de setpoint de enfriamiento y los sube a 25
                for i in range(1, len(sched.obj)):
                    try:
                        val = float(sched.obj[i])
                        if 20.0 <= val < 24.8:
                            sched.obj[i] = 25.0
                    except: pass

    # 3. REDUCCION DRASTICA DE CARGAS INTERNAS (luces, equipos, ocupacion)
    print("   -> Reduciendo luces y equipos al 40%...")
    # Luces
    if idf.idfobjects['LIGHTS']:
        for lt in idf.idfobjects['LIGHTS']:
            val = get_param_safe(lt, "Watts per Zone Floor Area")
            if val: set_param_safe(lt, "Watts per Zone Floor Area", float(val) * 0.4) # 40%

    # Equipos electricos (PCs, ascensores, etc.)
    if idf.idfobjects['ELECTRICEQUIPMENT']:
        for eq in idf.idfobjects['ELECTRICEQUIPMENT']:
            if 'elev' in eq.Name.lower(): # Ascensores desactivados
                set_param_safe(eq, "Design Level", 0.0)
                set_param_safe(eq, "Watts per Zone Floor Area", 0.0)
            else: # PCs normales
                val = get_param_safe(eq, "Watts per Zone Floor Area")
                if val: set_param_safe(eq, "Watts per Zone Floor Area", float(val) * 0.4) # 40%

    # 4. OCUPACION (Reducir densidad de personas)
    if idf.idfobjects['PEOPLE']:
        for pp in idf.idfobjects['PEOPLE']:
            val = get_param_safe(pp, "Zone Floor Area per Person")
            if val: set_param_safe(pp, "Zone Floor Area per Person", float(val) * 1.5) # Mayor area por persona = menor densidad

    # 5. INFILTRACION (Reducir fugas de aire al 50%)
    for inf in idf.idfobjects['ZONEINFILTRATION:DESIGNFLOWRATE']:
        val = get_param_safe(inf, "Flow per Exterior Surface Area")
        if val: set_param_safe(inf, "Flow per Exterior Surface Area", float(val) * 0.5)

    # 6. MATERIALES AISLANTES Y COP DE EQUIPOS DX
    for mat in idf.idfobjects['MATERIAL']:
        if any(x in mat.Name.lower() for x in ['insulation', 'wool', 'board']):
            set_param_safe(mat, "Conductivity", 1.5)
            set_param_safe(mat, "Specific Heat", 840)
    
    for coil in idf.idfobjects['COIL:COOLING:DX:TWOSPEED']:
        set_param_safe(coil, "High Speed Gross Rated Cooling COP", 4.0) 
        set_param_safe(coil, "Low Speed Gross Rated Cooling COP", 4.0)
    for coil in idf.idfobjects['COIL:COOLING:DX:SINGLESPEED']:
        set_param_safe(coil, "Gross Rated Cooling COP", 4.0)

    # 7. FERIADOS (Eliminar todos para asegurar dias laborales completos)
    while idf.idfobjects['RUNPERIODCONTROL:SPECIALDAYS']:
        idf.removeidfobject(idf.idfobjects['RUNPERIODCONTROL:SPECIALDAYS'][0])

def apply_calibration(idf):
    """Configura la ubicacion del sitio y aplica la reduccion agresiva."""
    # Ubicacion: Guayaquil, Ecuador
    if idf.idfobjects['SITE:LOCATION']:
        site = idf.idfobjects['SITE:LOCATION'][0]
        site.Name, site.Latitude, site.Longitude = "Guayaquil", -2.15, -79.88
        site.Time_Zone, site.Elevation = -5.0, 4.0
    
    # Aplicar las reducciones de consumo
    apply_aggressive_reduction(idf)
    return idf

def set_runperiod(idf):
    while idf.idfobjects['RUNPERIOD']: idf.removeidfobject(idf.idfobjects['RUNPERIOD'][0])
    rp = idf.newidfobject('RUNPERIOD')
    rp.Name, rp.Begin_Month, rp.Begin_Day_of_Month = "Semana_Enero_Comparativa", START_MONTH, START_DAY
    end = date(START_YEAR, START_MONTH, START_DAY) + timedelta(days=N_DAYS - 1)
    rp.End_Month, rp.End_Day_of_Month, rp.Day_of_Week_for_Start_Day = end.month, end.day, "Monday"
    
    # Desactivar feriados del archivo meteorologico para simular dias laborales completos
    set_param_safe(rp, "Use Weather File Holidays", "No") 
    set_param_safe(rp, "Use Weather File Daylight", "No")
    set_param_safe(rp, "Apply Weekend Holiday", "Yes")

def ensure_output_meter(idf):
    while idf.idfobjects['OUTPUT:METER']: idf.removeidfobject(idf.idfobjects['OUTPUT:METER'][0])
    meter = idf.newidfobject('OUTPUT:METER')
    meter.Key_Name, meter.Reporting_Frequency = METER_NAME, "Hourly"

def run_energyplus(idf_path, epw_path, outdir):
    """Ejecuta EnergyPlus con el IDF y EPW indicados."""
    print("="*60)
    print(f"EJECUTANDO SIMULACION... {outdir}")
    subprocess.run([str(EPLUS_EXE), "-r", "-w", str(epw_path), "-d", str(outdir), str(idf_path)], capture_output=True)

# ---------------------------
# 5. PROCESAMIENTO DE RESULTADOS
# ---------------------------
def process_results(outdir):
    """Lee los resultados de la simulacion, los compara con datos reales y genera graficos."""
    csv_path = outdir / "eplusout.csv"
    if not csv_path.exists(): raise FileNotFoundError("No se genero el CSV de resultados.")
    
    print("Procesando resultados...")
    df_sim = pd.read_csv(csv_path)
    cols = [c for c in df_sim.columns if METER_NAME.upper() in c.upper()]
    
    start_str = f"{START_YEAR}-{START_MONTH:02d}-{START_DAY:02d} 01:00"
    df_sim["Datetime"] = pd.date_range(start=start_str, periods=len(df_sim), freq="h")
    df_sim["kWh"] = df_sim[cols[0]] / 3_600_000.0
    
    # Ajuste de hora: la hora 24 de E+ se mapea a hora 0 del dia siguiente, se corrige restando 1s
    df_sim["AdjustedDate"] = (df_sim["Datetime"] - pd.Timedelta(seconds=1)).dt.date
    
    # Agrupar consumo simulado por dia
    daily_sim = df_sim.groupby("AdjustedDate")["kWh"].sum().reset_index()
    daily_sim.columns = ["Date", "Sim_kWh"]
    daily_sim["Day_Num"] = range(1, len(daily_sim) + 1)

    # Cargar datos reales desde CSV (potencia en kW cada 15 min)
    daily_real = pd.DataFrame()
    if REAL_DATA_FILE.exists():
        print("   -> Leyendo datos reales...")
        df_real = pd.read_csv(REAL_DATA_FILE)
        df_real["ts"] = pd.to_datetime(df_real["ts"])
        
        # Filtrar exactamente la semana correspondiente a la simulacion
        sim_start = pd.Timestamp(f"{START_YEAR}-{START_MONTH:02d}-{START_DAY:02d}")
        sim_end = sim_start + pd.Timedelta(days=7)
        
        mask = (df_real["ts"] >= sim_start) & (df_real["ts"] < sim_end)
        df_real_week = df_real.loc[mask].copy()
        
        if not df_real_week.empty:
            df_real_week["Date"] = df_real_week["ts"].dt.date
            daily_real = df_real_week.groupby("Date")["real_kw"].sum() * 0.25
            daily_real = daily_real.reset_index()
            daily_real.columns = ["Date", "Real_kWh"]
            daily_real["Day_Num"] = range(1, len(daily_real) + 1)

    # Generar comparacion y graficos
    plots_dir = outdir / "comparacion"
    plots_dir.mkdir(exist_ok=True)
    
    if not daily_real.empty:
        merged = pd.merge(daily_sim, daily_real, on="Day_Num", suffixes=("_Sim", "_Real"))
        merged["Dia"] = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes", "Sabado", "Domingo"][:len(merged)]
        
        csv_out = plots_dir / "Comparacion_Final.csv"
        merged.to_csv(csv_out, index=False)
        print(f"Dataset generado: {csv_out}")

        # Gráfico
        plt.figure(figsize=(10, 6))
        x = merged["Day_Num"]
        w = 0.35
        plt.bar(x - w/2, merged["Real_kWh"], w, label='Real (2023)', color='#2c3e50')
        plt.bar(x + w/2, merged["Sim_kWh"], w, label='Simulado (Ahorro)', color='#27ae60')
        plt.xticks(x, merged["Dia"])
        plt.ylabel("kWh Diario")
        plt.title("Comparacion Final: Simulacion vs Realidad")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / "Grafico_Barras_Final.png")
        print(f"Grafico guardado: {plots_dir / 'Grafico_Barras_Final.png'}")
        
        print("\n--- RESULTADOS LUNES ---")
        row = merged.iloc[0]
        print(f"Real: {row['Real_kWh']:.1f} kWh | Sim: {row['Sim_kWh']:.1f} kWh | Diff: {row['Sim_kWh'] - row['Real_kWh']:.1f}")

def main():
    if not IDF_ORIGINAL.exists(): return
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    outdir = OUTPUT_ROOT / f"Sim_Ahorro_{START_DAY}"
    outdir.mkdir(parents=True, exist_ok=True)

    clean_previous_failures()
    update_to_v23()
    try: IDF.setiddname(str(IDD_FILE))
    except: pass

    idf = IDF(str(IDF_TRABAJO), str(EPW_FILE))
    idf = apply_calibration(idf)
    set_runperiod(idf)
    ensure_output_meter(idf)
    
    idf_run = outdir / "in_calibrado_ahorro.idf"
    idf.saveas(str(idf_run))
    run_energyplus(idf_run, EPW_FILE, outdir)
    process_results(outdir)

if __name__ == "__main__":
    main()
