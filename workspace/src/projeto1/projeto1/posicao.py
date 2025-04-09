import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Header, Float64
from geometry_msgs.msg import Twist
import time
import numpy as np
import random

class NoDePosicao(Node):

    def __init__(self):
        super().__init__('Posicao')

        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.subscription_joint = self.create_subscription(
            JointState, '/joint_states', self.listener_callback_joint, qos_profile)
        
        self.subscription_laser = self.create_subscription(
            LaserScan, '/scan', self.listener_callback_laser, qos_profile)

        # Publisher
        self.publisher = self.create_publisher(JointState, '/joint_state', qos_profile)
        self.publisher_x = self.create_publisher(Float64, '/robot_x', qos_profile)
        self.publisher_cmd_vel = self.create_publisher(Twist, '/cmd_vel', qos_profile)

        self.timer = self.create_timer(0.5, self.update_position)

        # Constantes
        self.raio = 0.033 
        self.distancia_rodas = 0.178 

        # Variaveis
        self.posicao = [0.0, 0.0, 0.0]
        self.medidas = [None, None]  
        self.ultimas_medidas = [None, None]

        self.distancias = [0, 0]
        self.sigma_odometria = 0.2   
        self.sigma_lidar = 0.175   
        self.sigma_movimento = 0.002   

        self.jointL = 0.0
        self.jointR = 0.0

    def run(self):
        rclpy.spin(self)

    def listener_callback_joint(self, msg):
       
        self.jointL = msg.position[0]
        self.jointR = msg.position[1]

    def listener_callback_laser(self, msg):
     
        self.laser = msg.ranges[72] 

    def update_position(self):
 
        if self.ultimas_medidas[0] is None or self.ultimas_medidas[1] is None:
            self.ultimas_medidas[0] = self.jointL
            self.ultimas_medidas[1] = self.jointR
            return
        
       
        self.medidas[0] = self.jointL
        self.medidas[1] = self.jointR

        #
        diff_left = self.medidas[0] - self.ultimas_medidas[0]
        self.distancias[0] = diff_left * self.raio + random.random() * 0.002
        self.ultimas_medidas[0] = self.medidas[0]

        diff_right = self.medidas[1] - self.ultimas_medidas[1]
        self.distancias[1] = diff_right * self.raio + random.random() * 0.002
        self.ultimas_medidas[1] = self.medidas[1]

      
        deltaS = (self.distancias[0] + self.distancias[1]) / 2.0
        deltaTheta = (self.distancias[1] - self.distancias[0]) / self.distancia_rodas
        self.posicao[2] = (self.posicao[2] + deltaTheta) % (2 * np.pi)

       
        deltaSx = deltaS * np.cos(self.posicao[2])
        deltaSy = deltaS * np.sin(self.posicao[2])

        self.posicao[0] += deltaSx
        self.posicao[1] += deltaSy
        self.pub_x(self.posicao[0])


    def pub_x(self, x_position):
        
        msg = Float64()
        msg.data = x_position
        self.publisher_x.publish(msg)
        self.get_logger().info(f'X: {x_position}')

    def __del__(self):
        self.get_logger().info('Finalizando o nó! Tchau, tchau...')


def main(args=None):
    rclpy.init(args=args)
    node = NoDePosicao()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
