# simulation.py
import numpy as np
import itertools
import random
from config import elo_dict

def predecir_con_ajuste(elo_a, elo_b, model_coefs, model_intercept):
    
    # con el que entrenaste en main.py (elo_home, elo_away, diferencia_elo, es_neutral)
    features = np.array([elo_a, elo_b, (elo_a - elo_b), 1.0])
    
    # Cálculo logit
    log_odds = np.dot(model_coefs, features) + model_intercept
    
    # Aplicar Softmax para obtener probabilidades
    probs = np.exp(log_odds) / np.sum(np.exp(log_odds))
    return probs

def predecir_con_ajuste_suavizado(elo_a, elo_b, diff_suavizada, model_coefs, model_intercept
                                  ,eq_a_nombre, eq_b_nombre):
    # Usamos la diferencia suavizada en lugar de la 
    sesgos = {'Saudi Arabia': -0.5, 'Canada': -0.4}
    features = np.array([elo_a, elo_b, diff_suavizada, 1.0])
    
    log_odds = np.dot(model_coefs, features) + model_intercept

    if eq_a_nombre in sesgos:
        log_odds[2] += sesgos[eq_a_nombre]

    atenuacion = 1.3  # Valores entre 1.1 y 1.5 aplanan la curva
    probs = np.exp(log_odds / atenuacion) / np.sum(np.exp(log_odds / atenuacion))

    if diff_suavizada < -200: # El local es mucho más débil
        probs[2] = np.clip(probs[2], 0, 0.3) # Máximo 30% de probabilidad de ganar
        probs = probs / np.sum(probs)
        
    return probs

def simular_partido_con_ruido(eq_local, eq_visita, matriz_probs):
    par = tuple(sorted((eq_local, eq_visita)))
    return np.random.choice([0, 1, 2], p=matriz_probs[par])

def simular_fase_grupos_silenciosa(grupos_torneo, matriz_probs):
    clasificados_directos = []
    terceros_lugares = []
    for nombre_grupo, equipos in grupos_torneo.items():
        puntos = {equipo: 0 for equipo in equipos}
        for eq_local, eq_visita in itertools.combinations(equipos, 2):
            resultado = simular_partido_con_ruido(eq_local, eq_visita, matriz_probs)
            if resultado == 2: puntos[eq_local] += 3
            elif resultado == 1: puntos[eq_local] += 1; puntos[eq_visita] += 1
            else: puntos[eq_visita] += 3
        posiciones = sorted(puntos.items(), key=lambda x: (x[1], elo_dict.get(x[0], 1500)), reverse=True)
        clasificados_directos.extend([posiciones[0][0], posiciones[1][0]])
        terceros_lugares.append(posiciones[2])
    
    terceros_ordenados = sorted(terceros_lugares, key=lambda x: (x[1], elo_dict.get(x[0], 1500)), reverse=True)
    return clasificados_directos + [eq[0] for eq in terceros_ordenados[:8]]

def simular_partido_eliminatoria(equipo_a, equipo_b, matriz_probs):
    resultado = simular_partido_con_ruido(equipo_a, equipo_b, matriz_probs)
    if resultado == 2: return equipo_a
    elif resultado == 0: return equipo_b
    else: return random.choice([equipo_a, equipo_b])

def simular_mundial_completo(grupos_torneo, matriz_probs):
    ronda_actual = simular_fase_grupos_silenciosa(grupos_torneo, matriz_probs)
    while len(ronda_actual) > 1:
        siguiente_ronda = []
        for i in range(0, len(ronda_actual), 2):
            siguiente_ronda.append(simular_partido_eliminatoria(ronda_actual[i], ronda_actual[i+1], matriz_probs))
        ronda_actual = siguiente_ronda
    return ronda_actual[0]