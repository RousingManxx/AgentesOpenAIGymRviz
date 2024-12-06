#!/usr/bin/env python3

import rospy
import gymnasium as gym
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Configuración de parámetros de Q-learning
state_space = (20, 20, 20)  # Tres dimensiones: coseno, seno, velocidad angular
action_space = 5  # Cinco acciones para discretizar el espacio continuo de acciones en Pendulum
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 10000

# Inicialización de la tabla Q
q_table = np.zeros(state_space + (action_space,))

# Discretización del espacio de estados
def discretize_state(state):
    bins = [
        np.linspace(-1.0, 1.0, state_space[0]),  # Coseno del ángulo
        np.linspace(-1.0, 1.0, state_space[1]),  # Seno del ángulo
        np.linspace(-8.0, 8.0, state_space[2])   # Velocidad angular
    ]
    indices = [np.digitize(state[i], bins[i]) - 1 for i in range(len(state))]
    return tuple(np.clip(indices, 0, np.array(state_space) - 1))

# Crear marcador para visualizar el péndulo
def create_pendulum_marker(theta):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "pendulum"
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0

    pendulum_length = 1.0  # Longitud del péndulo
    x = pendulum_length * np.sin(theta)  # Coordenada X
    z = pendulum_length * np.cos(theta)  # Coordenada Z

    base = Point(0.0, 0.0, 0.0)
    end = Point(x, 0.0, z)

    marker.points.append(base)
    marker.points.append(end)
    return marker

# Entrenamiento del agente
env = gym.make('Pendulum-v1', render_mode=None)
rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    discretized_state = discretize_state(state)
    total_reward = 0
    done = False

    while not done:
        # Selección de acción (exploración/explotación)
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(0, action_space)
        else:
            action_idx = np.argmax(q_table[discretized_state])
        
        # Convertir acción discreta a continua
        action = np.linspace(-2.0, 2.0, action_space)[action_idx]

        # Ejecutar la acción
        next_state, reward, terminated, truncated, _ = env.step([action])
        next_discretized_state = discretize_state(next_state)

        # Actualización de Q-table
        best_next_action = np.argmax(q_table[next_discretized_state])
        q_table[discretized_state][action_idx] += learning_rate * (
            reward + discount_factor * q_table[next_discretized_state][best_next_action] - q_table[discretized_state][action_idx]
        )

        discretized_state = next_discretized_state
        total_reward += reward

        if terminated or truncated:
            done = True

    rewards.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards[-100:])
        print(f"Episodio {episode + 1}, Recompensa promedio: {avg_reward:.2f}")

np.save('q_table_pendulum.npy', q_table)
print("Entrenamiento completado.")

# Visualización en RViz
rospy.init_node('pendulum_visualization')
pendulum_publisher = rospy.Publisher('/pendulum/pendulum_marker', Marker, queue_size=10)

rate = rospy.Rate(10)

state, _ = env.reset()
discretized_state = discretize_state(state)

done = False
while not rospy.is_shutdown() and not done:
    # Seleccionar acción entrenada
    action_idx = np.argmax(q_table[discretized_state])
    action = np.linspace(-2.0, 2.0, action_space)[action_idx]

    # Ejecutar acción
    next_state, _, terminated, truncated, _ = env.step([action])
    discretized_state = discretize_state(next_state)

    # Publicar marcador del péndulo
    theta = np.arctan2(next_state[1], next_state[0])  # Ángulo del péndulo
    pendulum_marker = create_pendulum_marker(theta)
    pendulum_publisher.publish(pendulum_marker)

    done = terminated or truncated
    rate.sleep()

env.close()