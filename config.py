#config.py
import pandas as pd

# Diccionario de grupos inicial (puedes ampliarlo)
grupos_48 = {
    'Grp A': ['Argentina', 'Saudi Arabia', 'Mexico', 'Poland'],
    'Grp B': ['England', 'Iran', 'United States', 'Wales'],
    'Grp C': ['France', 'Australia', 'Denmark', 'Tunisia'],
    'Grp D': ['Spain', 'Costa Rica', 'Germany', 'Japan'],
    'Grp E': ['Belgium', 'Canada', 'Morocco', 'Croatia'],
    'Grp F': ['Brazil', 'Serbia', 'Switzerland', 'Cameroon'],
    'Grp G': ['Portugal', 'Ghana', 'Uruguay', 'South Korea'],
    'Grp H': ['Netherlands', 'Senegal', 'Ecuador', 'Qatar'],
    'Grp I': ['Italy', 'Sweden', 'Colombia', 'Chile'],
    'Grp J': ['Nigeria', 'Egypt', 'Algeria', 'Mali'],
    'Grp K': ['Peru', 'Paraguay', 'Venezuela', 'Bolivia'],
    'Grp L': ['Norway', 'Turkey', 'Romania', 'Uzbekistan']
}

# Inicializamos el diccionario de ELO vacío, se llenará al ejecutar el entrenamiento
elo_dict = {
    'Argentina': 2000.0,
    'Brazil': 1950.0,
    'France': 1980.0,
    'Germany' : 1910.0,
    'Spain': 1920.0,
    'England': 1900.0,
    'Portugal': 1880.0,
    'Saudi Arabia': 1550.0, # Subido para que el modelo no los humille
    'Iran': 1600.0
    }