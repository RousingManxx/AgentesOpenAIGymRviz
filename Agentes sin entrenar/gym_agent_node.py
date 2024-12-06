#!/usr/bin/env python3

# import rospy
# from visualization_msgs.msg import Marker, MarkerArray
# import numpy as np
# import gymnasium as gym

# class QLearningAgent:
#     def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
#         self.state_space = state_space
#         self.action_space = action_space
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.epsilon = epsilon
#         self.epsilon_decay = epsilon_decay
#         self.epsilon_min = epsilon_min
#         self.q_table = np.zeros(state_space + (action_space.n,))

#     def choose_action(self, state):
#         if np.random.rand() < self.epsilon:
#             return self.action_space.sample()
#         else:
#             return np.argmax(self.q_table[state])

#     def update_q_table(self, state, action, reward, next_state):
#         best_next_action = np.argmax(self.q_table[next_state])
#         self.q_table[state][action] += self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - self.q_table[state][action])

#     def decay_epsilon(self):
#         self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# def discretize_state(state):
#     bins = [np.linspace(-2.4, 2.4, num=24),
#             np.linspace(-2.0, 2.0, num=24),
#             np.linspace(-0.5, 0.5, num=24),
#             np.linspace(-0.5, 0.5, num=24)]
    
#     state_indices = [np.digitize(state[i], bins[i]) - 1 for i in range(len(state))]
#     return tuple(np.clip(state_indices, 0, 23))

# def publish_cart_position(cart_x):
#     marker_array = MarkerArray()
#     marker = Marker()
#     marker.header.frame_id = "world"
#     marker.type = Marker.CUBE
#     marker.action = Marker.ADD
#     marker.scale.x = 0.5
#     marker.scale.y = 0.1
#     marker.scale.z = 0.1
#     marker.color.a = 1.0
#     marker.color.r = 0.0
#     marker.color.g = 1.0
#     marker.color.b = 0.0
#     marker.pose.position.x = cart_x
#     marker.pose.position.y = 0.0
#     marker.pose.position.z = 0.0
#     marker_array.markers.append(marker)
#     cart_publisher.publish(marker_array)

# def publish_pole_angle(pole_angle, cart_x):
#     marker = Marker()
#     marker.header.frame_id = "world"
#     marker.type = Marker.CYLINDER
#     marker.action = Marker.ADD
#     marker.scale.x = 0.05
#     marker.scale.y = 0.05
#     marker.scale.z = 1.0
#     marker.color.a = 1.0
#     marker.color.r = 1.0
#     marker.color.g = 0.0
#     marker.color.b = 0.0
#     marker.pose.position.x = cart_x
#     marker.pose.position.y = 0.0
#     marker.pose.position.z = 0.5
#     marker.pose.orientation.z = np.sin(pole_angle / 2.0)
#     marker.pose.orientation.w = np.cos(pole_angle / 2.0)
#     pole_publisher.publish(marker)

# if __name__ == "__main__":
#     rospy.init_node('gym_agent_node')
#     cart_publisher = rospy.Publisher('/cartpole/cart_position', MarkerArray, queue_size=10)
#     pole_publisher = rospy.Publisher('/cartpole/pole_angle', Marker, queue_size=10)

#     env = gym.make('CartPole-v1')
#     agent = QLearningAgent(state_space=(24, 24, 24, 24), action_space=env.action_space)

#     for episode in range(1000):
#         observation = env.reset()[0]  # Obtiene directamente la observación inicial sin indexar
#         state = discretize_state(observation)
#         total_reward = 0
#         done = False

#         while not done:
#             action = agent.choose_action(state)
#             next_observation, reward, done, _, _ = env.step(action)
#             next_state = discretize_state(next_observation)  # Pasa la observación completa sin indexar
#             agent.update_q_table(state, action, reward, next_state)
#             state = next_state
#             total_reward += reward

#             # Extraer posición del carrito y ángulo del poste para visualización
#             cart_x = next_observation[0]  # Posición del carrito
#             pole_angle = next_observation[2]  # Ángulo del poste

#             # Publicar los datos para visualización en RViz
#             publish_cart_position(cart_x)
#             publish_pole_angle(pole_angle, cart_x)

#             rospy.sleep(0.05)

#         agent.decay_epsilon()
#         rospy.loginfo(f'Episode {episode} - Total Reward: {total_reward}')

#     env.close()


import rospy
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import gymnasium as gym
from geometry_msgs.msg import Point

class QLearningAgent:
    def __init__(self, state_space, action_space, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros(state_space + (action_space.n,))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (reward + self.discount_factor * self.q_table[next_state][best_next_action] - self.q_table[state][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def discretize_state(state):
    bins = [np.linspace(-2.4, 2.4, num=24),
            np.linspace(-2.0, 2.0, num=24),
            np.linspace(-0.5, 0.5, num=24),
            np.linspace(-0.5, 0.5, num=24)]
    
    state_indices = [np.digitize(state[i], bins[i]) - 1 for i in range(len(state))]
    return tuple(np.clip(state_indices, 0, 23))

def publish_cart_position(cart_x):
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = "world"  # Utilizar "world" como el frame de referencia
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.scale.x = 0.5
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.pose.position.x = cart_x
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0
    marker_array.markers.append(marker)
    cart_publisher.publish(marker_array)

def publish_pole_angle(pole_angle, cart_x):
    marker = Marker()
    marker.header.frame_id = "world"  # Asegúrate de que el marco de referencia sea el mismo
    marker.type = Marker.LINE_STRIP  # Usamos LINE_STRIP para dibujar una línea entre el carrito y el poste
    marker.action = Marker.ADD
    marker.scale.x = 0.05  # Grosor de la línea
    marker.color.a = 1.0  # Transparencia
    marker.color.r = 1.0  # Color rojo
    marker.color.g = 0.0  # Sin verde
    marker.color.b = 0.0  # Sin azul

    # Posición inicial del marcador (el carrito)
    marker.points = []
    start_point = Point()
    start_point.x = cart_x  # La posición del carrito
    start_point.y = 0.0  # En el eje Y el carrito está centrado
    start_point.z = 0.0  # Altura del carrito desde el suelo

    # Posición final del marcador (la punta del poste)
    end_point = Point()
    end_point.x = cart_x + np.sin(pole_angle)  # Desplazamiento en X por el ángulo
    end_point.y = np.cos(pole_angle) - 1  # Desplazamiento en Y por el ángulo
    end_point.z = 1.0  # Altura de la punta del poste (ajustable)

    # Añadir los puntos al marcador
    marker.points.append(start_point)
    marker.points.append(end_point)

    # Publicar el marcador
    pole_publisher.publish(marker)

if __name__ == "__main__":
    rospy.init_node('gym_agent_node')
    cart_publisher = rospy.Publisher('/cartpole/cart_position', MarkerArray, queue_size=10)
    pole_publisher = rospy.Publisher('/cartpole/pole_angle', Marker, queue_size=10)

    env = gym.make('CartPole-v1')
    agent = QLearningAgent(state_space=(24, 24, 24, 24), action_space=env.action_space)

    for episode in range(1000):
        observation = env.reset()[0]  # Obtiene directamente la observación inicial sin indexar
        state = discretize_state(observation)
        total_reward = 0
        done = False

        while not done:
            action = agent.choose_action(state)
            next_observation, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_observation)  # Pasa la observación completa sin indexar
            agent.update_q_table(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            # Extraer posición del carrito y ángulo del poste para visualización
            cart_x = next_observation[0]  # Posición del carrito
            pole_angle = next_observation[2]  # Ángulo del poste

            # Publicar los datos para visualización en RViz
            publish_cart_position(cart_x)
            publish_pole_angle(pole_angle, cart_x)

            rospy.sleep(0.05)

        agent.decay_epsilon()
        rospy.loginfo(f'Episode {episode} - Total Reward: {total_reward}')

    env.close()