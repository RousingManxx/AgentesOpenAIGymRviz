#!/usr/bin/env python3

import gym
import rospy
import numpy as np
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

# Inicializar el nodo de ROS
rospy.init_node('acrobot_agent')

# Crear el entorno de Gym
env = gym.make('Acrobot-v1')
state = env.reset()

# Crear publicadores de ROS para los topics de RViz
link1_pub = rospy.Publisher('/acrobot/link1', Marker, queue_size=10)
link2_pub = rospy.Publisher('/acrobot/link2', Marker, queue_size=10)
arm_pub = rospy.Publisher('/acrobot/arm', Marker, queue_size=10)
rate = rospy.Rate(30)  # Frecuencia de publicación en Hz

# Función para crear un marcador de RViz
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

# Función para crear un marcador de línea para el brazo
def create_arm_marker(start_pos, end_pos):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05  # Grosor de la línea
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.points = [Point(*start_pos), Point(*end_pos)]
    marker.id = 3
    return marker

# Bucle principal del agente (se ejecutará indefinidamente hasta que lo detengas manualmente)
while not rospy.is_shutdown():
    # Seleccionar una acción aleatoria
    action = env.action_space.sample()

    # Ejecutar la acción en el entorno
    next_state, reward, terminated, truncated, info = env.step(action)

    # Calcular las posiciones de los enlaces del Acrobot
    # Longitud de las barras del Acrobot
    l1, l2 = 1.0, 1.0
    link1_pos = [l1 * np.cos(next_state[0]), l1 * np.sin(next_state[0]), 0.0]
    link2_pos = [
        link1_pos[0] + l2 * np.cos(next_state[0] + next_state[1]),
        link1_pos[1] + l2 * np.sin(next_state[0] + next_state[1]),
        0.0
    ]

    # Crear y publicar los marcadores en RViz
    marker1 = create_marker(link1_pos, marker_id=1, color=(1.0, 0.0, 0.0, 1.0))
    marker2 = create_marker(link2_pos, marker_id=2, color=(0.0, 0.0, 1.0, 1.0))
    arm_marker = create_arm_marker([0.0, 0.0, 0.0], link1_pos)
    arm_marker.points.append(Point(*link2_pos))
    link1_pub.publish(marker1)
    link2_pub.publish(marker2)
    arm_pub.publish(arm_marker)

    # Esperar según la frecuencia establecida
    rate.sleep()

# Cerrar el entorno cuando se detiene el nodo
env.close()