# main.py
import numpy as np
import itertools
from collections import Counter
from config import grupos_48, elo_dict
from simulation import predecir_con_ajuste, predecir_con_ajuste_suavizado, simular_mundial_completo
import joblib


try:
    modelo_final = joblib.load('modelo_entrenado.pkl')
    
    # Verificamos si es un Pipeline o el modelo directo
    if hasattr(modelo_final, 'named_steps'):
        # Si es un pipeline, buscamos el paso de regresión
        log_reg_step = modelo_final.named_steps['logisticregression']
    else:
        # Si no tiene steps, asumimos que el modelo es la regresión misma
        log_reg_step = modelo_final
    
    model_coefs = log_reg_step.coef_[0]
    model_intercept = log_reg_step.intercept_
    
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"¡ERROR al cargar o procesar el modelo: {e}")
    exit()

# --- PASO 2: PRE-CÁLCULO ---
todos_equipos = list(set([e for g in grupos_48.values() for e in g]))
matriz_probs = {}
for eqA, eqB in itertools.combinations(todos_equipos, 2):
    elo_a_ajustado = elo_dict.get(eqA, 1500) + np.random.normal(0, 60)
    elo_b_ajustado = elo_dict.get(eqB, 1500) + np.random.normal(0, 60)

    diff_suavizada = (elo_a_ajustado - elo_b_ajustado) / 1.0
    matriz_probs[tuple(sorted((eqA, eqB)))] = predecir_con_ajuste_suavizado(elo_a_ajustado,
                elo_b_ajustado, diff_suavizada, model_coefs, model_intercept, eqA, eqB)

# --- PASO 3: BUCLE DE SIMULACIÓN ---
N_SIMULACIONES = 10000
lista_campeones = []
for i in range(N_SIMULACIONES):
    lista_campeones.append(simular_mundial_completo(grupos_48, matriz_probs))

# --- PASO 4: RESULTADOS ---
conteo = Counter(lista_campeones)
for equipo, victorias in conteo.most_common(10):
    print(f"{equipo}: {(victorias/N_SIMULACIONES)*100:.1f}%")
    