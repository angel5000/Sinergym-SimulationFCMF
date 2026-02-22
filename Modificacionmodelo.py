import os
import json

print("\n" + "="*80)
print("PREPARACIÓN GLOBAL DEL MODELO DE EDIFICIO (epJSON)")
print("="*80 + "\n")

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Cambia esto por la ruta de tu modelo original exportado de OpenStudio/DesignBuilder
ARCHIVO_ORIGINAL = '/home/vboxuser/Descargas/ASHRAE901_OfficeMedium_STD2019_Miami.epJSON'
# Este será el archivo limpio que usarán tus scripts de Sinergym
ARCHIVO_LISTO = '/home/vboxuser/Descargas/EDIFICIO_SOLIDO.epJSON'

def preparar_modelo(input_path, output_path):
    if not os.path.exists(input_path):
        print(f" No se encontró el archivo original: {input_path}")
        return

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print("✔️ Archivo cargado correctamente. Aplicando modificaciones...")

        # 1. ACTUALIZACIÓN DE VERSIÓN (Compatibilidad Sinergym E+ 23.1)
        if "Version" in data:
            for key in data["Version"]:
                data["Version"][key]["version_identifier"] = "23.1"
                print("   - Versión actualizada a 23.1")

        # 2. ESTABILIDAD TERMODINÁMICA
        if "Building" in data:
            for key in data["Building"]:
                # Aumenta la tolerancia de convergencia para evitar errores por variaciones mínimas de temperatura
                data["Building"][key]["temperature_convergence_tolerance_value"] = 0.5
                print("   - Tolerancia de convergencia de temperatura fijada en 0.5")

        # 3. CONVERGENCIA HVAC
        if "SimulationControl" in data:
            for key in data["SimulationControl"]:
                # Da más intentos al sistema HVAC para resolver las ecuaciones térmicas antes de crashear
                data["SimulationControl"][key]["maximum_hvac_iterations"] = 60
                print("   - Iteraciones máximas de HVAC aumentadas a 60")

        # 4. SILENCIADOR DE REPORTES MASIVOS (Fix Anticolapso)
        # Se borran las peticiones de salidas innecesarias que congelan la terminal al final de la simulación
        claves_a_borrar = [
            "Output:Variable", 
            "Output:Meter", 
            "Output:Meter:MeterFileOnly",
            "Output:Table:SummaryReports", 
            "Output:SQLite"
        ]
        
        eliminados = 0
        for clave in claves_a_borrar:
            if clave in data:
                del data[clave]
                eliminados += 1
        if eliminados > 0:
            print(f"   - Silenciador activado: {eliminados} bloques de reportes eliminados.")

        # Guardar archivo optimizado
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        print(f"\n Modelo optimizado guardado con éxito en: \n   {output_path}")

    except Exception as e:
        print(f"Error crítico procesando el JSON: {e}")

if __name__ == '__main__':
    preparar_modelo(ARCHIVO_ORIGINAL, ARCHIVO_LISTO)
    print("="*80)