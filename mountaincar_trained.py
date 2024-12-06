#!/usr/bin/env python3

import rospy
import gymnasium as gym
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

# Configuración de parámetros de Q-learning
state_space = (20, 20)  # Más granularidad para la discretización del espacio de estados
action_space = 3  # Tres acciones disponibles en MountainCar
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000

# Inicialización de la tabla Q
q_table = np.zeros(state_space + (action_space,))

# Discretización del espacio de estados
def discretize_state(state):
    bins = [
        np.linspace(-1.2, 0.6, state_space[0]),  # Posición
        np.linspace(-0.07, 0.07, state_space[1])  # Velocidad
    ]
    indices = [np.digitize(state[i], bins[i]) - 1 for i in range(len(state))]
    return tuple(np.clip(indices, 0, np.array(state_space) - 1))

# Crear marcador para el carro
def create_car_marker(position):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "mountaincar"
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = position[0]
    marker.pose.position.y = 0.0
    marker.pose.position.z = np.sin(3 * position[0])
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1
    marker.scale.y = 0.05
    marker.scale.z = 0.05
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    return marker

# Crear marcador para la carretera
def create_track_marker():
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "mountaincar"
    marker.id = 1
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.02
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    for x in np.linspace(-1.2, 0.6, 100):
        y = np.sin(3 * x)
        point = Point()
        point.x = x
        point.y = 0.0
        point.z = y
        marker.points.append(point)

    return marker

# Entrenamiento del agente
env = gym.make('MountainCar-v0')
rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    discretized_state = discretize_state(state)
    total_reward = 0
    done = False

    while not done:
        # Selección de acción (exploración/explotación)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[discretized_state])

        # Ejecución de la acción
        next_state, reward, terminated, truncated, _ = env.step(action)
        next_discretized_state = discretize_state(next_state)

        # Actualización de Q-table
        best_next_action = np.argmax(q_table[next_discretized_state])
        q_table[discretized_state][action] += learning_rate * (
            reward + discount_factor * q_table[next_discretized_state][best_next_action] - q_table[discretized_state][action]
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

np.save('q_table_mountaincar.npy', q_table)
print("Entrenamiento completado.")

# Publicación en RViz
rospy.init_node('mountaincar_visualization')
car_publisher = rospy.Publisher('/mountaincar/car_position', Marker, queue_size=10)
track_publisher = rospy.Publisher('/mountaincar/track', Marker, queue_size=10)

rate = rospy.Rate(10)

track_marker = create_track_marker()

state, _ = env.reset()
discretized_state = discretize_state(state)

done = False
while not rospy.is_shutdown() and not done:
    action = np.argmax(q_table[discretized_state])
    state, _, terminated, truncated, _ = env.step(action)
    discretized_state = discretize_state(state)

    # Publicar el marcador del carro
    car_marker = create_car_marker([state[0]])
    car_publisher.publish(car_marker)

    # Publicar el marcador de la carretera en cada iteración
    track_publisher.publish(track_marker)

    done = terminated or truncated
    rate.sleep()

env.close()
