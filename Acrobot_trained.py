#!/usr/bin/env python3

import time
import gym
import rospy
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

# Parámetros de Q-learning
state_space = (6, 6, 6, 6, 6, 6)  # Ajustar la discretización del espacio de estados
action_space = 3
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
num_episodes = 1000

# Inicialización de la tabla Q
q_table = np.zeros(state_space + (action_space,))

# Función para discretizar el estado
def discretize_state(state):
    bins = [
        np.linspace(-1.0, 1.0, num=state_space[0]),  # cos(theta1)
        np.linspace(-1.0, 1.0, num=state_space[1]),  # sin(theta1)
        np.linspace(-1.0, 1.0, num=state_space[2]),  # cos(theta2)
        np.linspace(-1.0, 1.0, num=state_space[3]),  # sin(theta2)
        np.linspace(-4.0, 4.0, num=state_space[4]),  # theta1_dot
        np.linspace(-9.0, 9.0, num=state_space[5])   # theta2_dot
    ]
    indices = [np.digitize(state[i], bins[i]) - 1 for i in range(len(state))]
    return tuple(np.clip(indices, 0, np.array(state_space) - 1))

# Función para crear marcadores de RViz
def create_marker(position, marker_id, color):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position = Point(*position)
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.1
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
    marker.id = marker_id
    return marker

def create_arm_marker(start_pos, end_pos):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.points = [Point(*start_pos), Point(*end_pos)]
    return marker

# Entrenamiento del agente
env = gym.make('Acrobot-v1')
episode_rewards = []

print("Entrenando agente...")

for episode in range(num_episodes):
    obs, _ = env.reset()  # Actualizado para obtener el estado inicial
    state = discretize_state(obs)
    total_reward = 0
    done = False

    while not done:
        # Seleccionar una acción (exploración/explotación)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Realizar la acción
        obs, reward, terminated, truncated, info = env.step(action)  # Actualizado
        next_state = discretize_state(obs)
        best_next_action = np.argmax(q_table[next_state])

        # Actualizar la tabla Q
        q_table[state][action] += learning_rate * (
            reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action]
        )
        state = next_state
        total_reward += reward

        if terminated or truncated:
            done = True

    # Reducir epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Guardar la recompensa total
    episode_rewards.append(total_reward)

    # Mostrar el progreso cada 100 episodios
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"Episodio: {episode + 1}, Recompensa Promedio: {avg_reward:.2f}")

# Guardar la tabla Q entrenada
np.save('q_table_acrobot.npy', q_table)
print("Entrenamiento completado. Visualizando agente entrenado en RViz...")

# Publicadores de RViz
rospy.init_node('acrobot_agent')
link1_pub = rospy.Publisher('/acrobot/link1', Marker, queue_size=10)
link2_pub = rospy.Publisher('/acrobot/link2', Marker, queue_size=10)
arm_pub = rospy.Publisher('/acrobot/arm', Marker, queue_size=10)
rate = rospy.Rate(30)

# Evaluación y visualización del agente entrenado
obs, _ = env.reset()  # Actualizado
state = discretize_state(obs)
done = False

while not rospy.is_shutdown() and not done:
    action = np.argmax(q_table[state])
    next_state, reward, terminated, truncated, _ = env.step(action)
    next_state = discretize_state(next_state)

    # Posiciones del Acrobot
    l1, l2 = 1.0, 1.0
    link1_pos = [l1 * np.sin(env.state[0]), -l1 * np.cos(env.state[0]), 0.0]
    link2_pos = [
        link1_pos[0] + l2 * np.sin(env.state[0] + env.state[1]),
        link1_pos[1] - l2 * np.cos(env.state[0] + env.state[1]),
        0.0
    ]

    marker1 = create_marker(link1_pos, marker_id=1, color=(1.0, 0.0, 0.0, 1.0))
    marker2 = create_marker(link2_pos, marker_id=2, color=(0.0, 0.0, 1.0, 1.0))
    arm_marker = create_arm_marker([0.0, 0.0, 0.0], link1_pos)
    arm_marker.points.append(Point(*link2_pos))
    link1_pub.publish(marker1)
    link2_pub.publish(marker2)
    arm_pub.publish(arm_marker)

    state = next_state
    done = terminated or truncated
    rate.sleep()

env.close()