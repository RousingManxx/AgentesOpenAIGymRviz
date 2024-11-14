#!/usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import gymnasium as gym
import numpy as np

# Inicializar el nodo de ROS
rospy.init_node('mountaincar_visualizer')

# Crear los publicadores
car_publisher = rospy.Publisher('/mountaincar/car_position', Marker, queue_size=10)
track_publisher = rospy.Publisher('/mountaincar/track', Marker, queue_size=10)

# Crear el entorno de MountainCar
env = gym.make('MountainCar-v0')
env = env.unwrapped  # Desempaquetar el entorno base para acceder a los límites
min_position = env.min_position
max_position = env.max_position

# Configurar y publicar el marcador del carro
def publish_car(position):
    car_marker = Marker()
    car_marker.header.frame_id = "world"
    car_marker.header.stamp = rospy.Time.now()
    car_marker.ns = "mountaincar"
    car_marker.id = 0
    car_marker.type = Marker.CUBE
    car_marker.action = Marker.ADD
    car_marker.scale.x = 0.1
    car_marker.scale.y = 0.05
    car_marker.scale.z = 0.05
    car_marker.color.a = 1.0
    car_marker.color.r = 1.0
    car_marker.color.g = 0.0
    car_marker.color.b = 0.0
    car_marker.pose.position.x = position
    car_marker.pose.position.y = 0.0
    car_marker.pose.position.z = np.sin(3 * position)  # Alinear el carro con la carretera
    car_marker.pose.orientation.x = 0.0
    car_marker.pose.orientation.y = 0.0
    car_marker.pose.orientation.z = 0.0
    car_marker.pose.orientation.w = 1.0
    car_publisher.publish(car_marker)

# Configurar y publicar el marcador de la carretera
def publish_track():
    track_marker = Marker()
    track_marker.header.frame_id = "world"
    track_marker.header.stamp = rospy.Time.now()
    track_marker.ns = "mountaincar"
    track_marker.id = 1
    track_marker.type = Marker.LINE_STRIP
    track_marker.action = Marker.ADD
    track_marker.scale.x = 0.02
    track_marker.color.a = 1.0
    track_marker.color.r = 0.0
    track_marker.color.g = 1.0
    track_marker.color.b = 0.0

    for x in np.linspace(min_position, max_position, 100):
        y = np.sin(3 * x)
        point = Point()
        point.x = x
        point.y = 0.0
        point.z = y  # Establecer la carretera en el eje Z
        track_marker.points.append(point)

    track_publisher.publish(track_marker)

# Simulación principal
rate = rospy.Rate(10)  # 10 Hz
state, _ = env.reset()  # Reiniciar el entorno
done = False

while not rospy.is_shutdown() and not done:
    # Acción aleatoria solo para visualización; puedes cambiarla por tu agente
    action = env.action_space.sample()
    next_state, reward, done, _, _ = env.step(action)

    # Publicar la posición del carro alineada con la carretera
    publish_car(next_state[0])

    # Publicar la carretera (una sola vez es suficiente)
    publish_track()

    rate.sleep()