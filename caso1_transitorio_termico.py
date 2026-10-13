import numpy as np
import matplotlib.pyplot as plt

# Configuración de estilo académico ("Nivel PhD" / Listas para Publicación)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 300
})

def resolver_transitorio_termico(dt: float, t_final: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Resuelve la EDO del transitorio térmico de un motor usando el Método de Euler.
    
    Parámetros:
    -----------
    dt : float
        Paso de tiempo de integración [s].
    t_final : float
        Tiempo total de la simulación [s].
        
    Retorna:
    --------
    t : np.ndarray
        Vector de tiempo.
    T : np.ndarray
        Vector de temperaturas estimadas [°C].
    """
    # Parámetros físicos del motor de 15 kW
    T_amb = 25.0        # Temperatura ambiente [°C]
    P_perdidas = 1200.0 # Pérdidas térmicas constantes [W]
    C_th = 45000.0      # Capacidad térmica [J/°C]
    tau = 1200.0        # Constante de tiempo térmica [s]
    
    # Inicialización de vectores
    t = np.arange(0, t_final + dt, dt)
    T = np.zeros(len(t))
    T[0] = T_amb # Condición inicial
    
    # Bucle del método de Euler hacia adelante
    for i in range(0, len(t) - 1):
        dT_dt = (P_perdidas / C_th) - (1.0 / tau) * (T[i] - T_amb)
        T[i+1] = T[i] + dT_dt * dt
        
    return t, T

# --- Configuración del Experimento Numérico ---
t_final_sim = 3600.0  # 1 hora de operación
pasos_tiempo = [120.0, 60.0, 10.0] # Diferentes delta t para evaluar el error

# Parámetros para la solución exacta continua
T_amb, P_perdidas, C_th, tau = 25.0, 1200.0, 45000.0, 1200.0
t_exacto = np.linspace(0, t_final_sim, 1000)
T_exacto = T_amb + (tau * P_perdidas / C_th) * (1.0 - np.exp(-t_exacto / tau))

# --- Construcción del Gráfico de Alta Calidad ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Gráfico Principal: Soluciones numéricas vs Exacta
ax1.plot(t_exacto / 60.0, T_exacto, label="Solución Exacta (Analítica)", color="black", linewidth=2.0)

estilos_linea = [('--', 'o'), (':', 's'), ('-.', '^')]
for dt, (estilo, marcador) in zip(pasos_tiempo, estilos_linea):
    t_num, T_num = resolver_transitorio_termico(dt, t_final_sim)
    
    # Graficar curvas numéricas
    ax1.plot(t_num / 60.0, T_num, linestyle=estilo, label=f"Euler ($\Delta t = {dt}$ s)", alpha=0.85)
    
    # Calcular error absoluto en los puntos coincidentes con la solución exacta
    T_exacta_coincidente = T_amb + (tau * P_perdidas / C_th) * (1.0 - np.exp(-t_num / tau))
    error_abs = np.abs(T_exacta_coincidente - T_num)
    
    # Gráfico Secundario: Evolución del Error de Truncamiento
    ax2.plot(t_num / 60.0, error_abs, linestyle=estilo, label=f"Error ($\Delta t = {dt}$ s)")

# Ajustes de formato - Panel 1
ax1.set_title("Evolución de la Temperatura en el Estátor")
ax1.set_xlabel("Tiempo (minutos)")
ax1.set_ylabel("Temperatura ($^{\circ}$C)")
ax1.grid(True)
ax1.legend(loc="lower right")

# Ajustes de formato - Panel 2
ax2.set_title("Magnitud del Error de Truncamiento Global")
ax2.set_xlabel("Tiempo (minutos)")
ax2.set_ylabel("Error Absoluto ($^{\circ}$C)")
ax2.grid(True)
ax2.legend(loc="upper left")

plt.tight_layout()
plt.show()
