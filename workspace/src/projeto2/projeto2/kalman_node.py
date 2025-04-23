import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import tf_transformations
from std_msgs.msg import Header
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
import random
import math
import numpy as np

class kalman_node(Node):
    def __init__(self):
        super().__init__('posicao')

        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.subscription_odom = self.create_subscription(Odometry, '/odom', self.listener_callback_odom, qos_profile)
        # Publishers
        self.velocidade_ = self.create_publisher(Twist, "/cmd_vel", 10)
        self.publisher_posicao = self.create_publisher(Pose2D, '/posicao', qos_profile)

        #self.timer = self.create_timer(0.01, self.update) 

        # Variáveis de estado
        self.raio = 0.033
        self.distancia_rodas = 0.178
        self.pose = [0.0, 0.0, 0.0]  # x, y, theta

        # Ruídos
      
        self.sigma_x = 0.001  # era 0.005
        self.sigma_y = 0.001  # era 0.005
        self.sigma_z = 0.001  # não usado diretamente, mas reduzido também
        self.sigma_th = math.radians(0.005)  # era 0.01 rad
        self.sigma_v = 0.0005  # era 0.001
        self.sigma_w = math.radians(0.005)  # era 0.01 rad

        self.sigma_z_x = 0.005  # era 0.01
        self.sigma_z_y = 0.005  # era 0.01

        self.v = 0.5
        self.raio_circ = .5
        self.w = self.v / self.raio_circ
        self.ct = self.get_clock().now().nanoseconds
        self.lt = self.ct
        self.dt = 0.0
    def velocidade(self):
        twist = Twist()
    
        twist.linear.x = self.v
        twist.angular.z = self.w

        self.velocidade_.publish(twist)

    def listener_callback_odom(self, msg):
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        _, _, self.yaw = tf_transformations.euler_from_quaternion([x, y, z, w])

        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y

        self.ct = self.get_clock().now().nanoseconds
        self.dt = (self.ct - self.lt) * 1e-9
        self.lt = self.ct
        self.update()

    def update(self):
        # 1. Publica velocidade
        self.velocidade()
        # 2. Simulação do movimento real com ruído (pose real "verdadeira" que o GPS veria)
        ksi_x = self.sigma_x * random.gauss(0, 1)
        ksi_y = self.sigma_y * random.gauss(0, 1)
        ksi_th = self.sigma_th * random.gauss(0, 1)
        ksi_v = self.sigma_v * random.gauss(0, 1)
        ksi_w = self.sigma_w * random.gauss(0, 1)

        Pv = self.pose.copy()
        Pv[0] += (self.v + ksi_v) * math.cos(Pv[2] + ksi_th) * self.dt + ksi_x
        Pv[1] += (self.v + ksi_v) * math.sin(Pv[2] + ksi_th) * self.dt + ksi_y
        Pv[2] += (self.w + ksi_w) * self.dt + ksi_th

        # 3. Simulação da medida (ex: GPS)
        C = np.array([[1, 0, 0], [0, 1, 0]])
        R = np.array([[self.sigma_z_x**2, 0], [0, self.sigma_z_y**2]])
        ruido = np.dot(np.linalg.cholesky(R), np.random.randn(2, 1)).flatten()
        y = np.dot(C, Pv) + ruido

        # 4. EKF: Previsão com base na pose anterior
        self.Pe = self.pose.copy()
        self.Pe[0] += self.v * math.cos(self.Pe[2]) * self.dt
        self.Pe[1] += self.v * math.sin(self.Pe[2]) * self.dt
        self.Pe[2] += self.w * self.dt

        # Matriz de covariância do processo (incerteza no movimento)
        Q = np.diag([self.sigma_x**2, self.sigma_y**2, self.sigma_th**2])
        R = np.diag([self.sigma_z_x**2, self.sigma_z_y**2])  # Medição

        F = np.array([
            [1, 0, -self.v * math.sin(self.Pe[2]) * self.dt],
            [0, 1,  self.v * math.cos(self.Pe[2]) * self.dt],
            [0, 0, 1]
        ])

        H = C

        # Covariância de erro (poderia ser persistente, mas aqui é estática para simplificar)
        P = Q

        z = np.dot(H, self.Pe)  # medida esperada
        K = np.dot(P, H.T).dot(np.linalg.inv(H.dot(P).dot(H.T) + R))  # ganho de Kalman
        self.Pe = self.Pe + np.dot(K, (y - z))  # correção

        self.pose = self.Pe
        self.publicar_posicao()
        
    def publicar_posicao(self):
        msg = Pose2D()
        msg.x = self.Pe[0]
        msg.y = self.Pe[1]
        msg.theta = self.Pe[2]
        self.publisher_posicao.publish(msg)
        self.get_logger().info(f'Publicando pose -> x: {msg.x:.2f}, y: {msg.y:.2f}, theta: {math.degrees(msg.theta):.2f}°')

    def __del__(self):
        self.get_logger().info('SAISAISAISAISAISIASAI')
        

def main(args=None):
    rclpy.init(args=args)
    node = kalman_node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
