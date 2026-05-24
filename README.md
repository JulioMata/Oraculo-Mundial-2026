# Oráculo del Balón 2026

Simulador de Monte Carlo para predecir los resultados de la Copa del Mundo 2026 utilizando modelos de Machine Learning (XGBoost/Regresión Logística) y Ranking ELO.

## Estructura del Proyecto
- `config.py`: Variables globales, diccionarios de equipos y configuración inicial.
- `simulation.py`: Lógica del torneo (fase de grupos y eliminatorias).
- `main.py`: Script principal para ejecutar las 10,000 simulaciones.

## Instalación
Asegúrate de tener instaladas las librerías necesarias:
```bash
pip install pandas numpy scikit-learn