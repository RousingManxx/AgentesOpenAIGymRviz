#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import gymnasium as gym
import numpy as np

# Inicializar el nodo de ROS
rospy.init_node('pendulum_visualizer')

# Crear publicadores
pendulum_publisher = rospy.Publisher('/pendulum/pendulum_marker', Marker, queue_size=10)
tip_publisher = rospy.Publisher('/pendulum/pendulum_tip', Marker, queue_size=10)

# Crear el entorno de Pendulum
env = gym.make('Pendulum-v1')
env = env.unwrapped  # Desempaquetar el entorno para un control más directo

# Configurar y publicar el marcador del péndulo
def publish_pendulum(theta):
    # Publicar el marcador de la línea del péndulo
    pendulum_marker = Marker()
    pendulum_marker.header.frame_id = "world"
    pendulum_marker.header.stamp = rospy.Time.now()
    pendulum_marker.ns = "pendulum"
    pendulum_marker.id = 0
    pendulum_marker.type = Marker.LINE_STRIP
    pendulum_marker.action = Marker.ADD
    pendulum_marker.scale.x = 0.05  # Grosor de la línea
    pendulum_marker.color.a = 1.0
    pendulum_marker.color.r = 0.0
    pendulum_marker.color.g = 1.0
    pendulum_marker.color.b = 0.0

    # Asegurarse de que la gravedad está dirigida hacia abajo en el eje z
    pendulum_length = 1.0  # Longitud del péndulo
    x = pendulum_length * np.sin(theta)  # Movimiento en el plano XY
    z = -pendulum_length * np.cos(theta)

    # Crear puntos para la línea del péndulo
    base = Point()
    base.x = 0.0
    base.y = 0.0
    base.z = 0.0

    end = Point()
    end.x = x
    end.y = 0.0  # No hay movimiento en el eje Y, solo X y Z
    end.z = z

    pendulum_marker.points.append(base)
    pendulum_marker.points.append(end)

    # Publicar el marcador de la línea
    pendulum_publisher.publish(pendulum_marker)

    # Publicar el marcador del círculo en la punta del péndulo
    tip_marker = Marker()
    tip_marker.header.frame_id = "world"
    tip_marker.header.stamp = rospy.Time.now()
    tip_marker.ns = "pendulum_tip"
    tip_marker.id = 1
    tip_marker.type = Marker.SPHERE
    tip_marker.action = Marker.ADD
    tip_marker.scale.x = 0.1  # Radio del círculo
    tip_marker.scale.y = 0.1
    tip_marker.scale.z = 0.1
    tip_marker.color.a = 1.0
    tip_marker.color.r = 1.0
    tip_marker.color.g = 0.0
    tip_marker.color.b = 0.0

    # Posición del círculo (punta del péndulo)
    tip_marker.pose.position.x = x
    tip_marker.pose.position.y = 0.0
    tip_marker.pose.position.z = z

    # Publicar el marcador del círculo
    tip_publisher.publish(tip_marker)

# Simulación principal
rate = rospy.Rate(10)  # 10 Hz
state, _ = env.reset()  # Reiniciar el entorno
done = False

while not rospy.is_shutdown() and not done:
    # Acción aleatoria solo para visualización; puedes cambiarla por tu agente
    action = env.action_space.sample()
    next_state, reward, done, _, _ = env.step(action)

    # Publicar la posición del péndulo
    theta = next_state[0]  # Ángulo del péndulo
    publish_pendulum(theta)

    rate.sleep()