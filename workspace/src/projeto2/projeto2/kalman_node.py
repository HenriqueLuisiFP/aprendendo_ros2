import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import tf_transformations
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
import random
import math
import numpy as np

class kalmannode(Node):
    def __init__(self):
        super().__init__('kalman')

        qos_profile = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        # Subscribers
        self.odom_subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile)

        self.laser_subscriber = self.create_subscription(LaserScan, '/scan', self.laser_callback, qos_profile)

        # Publishers
        self.laser_publisher = self.create_publisher(LaserScan, '/laser_data', qos_profile)
        self.pose_publisher = self.create_publisher(Pose2D, '/pose', qos_profile)

        self.timer = self.create_timer(0.1, self.update)
        
        #Parameters
        self.wheel_radius = 0.033
        self.wheel_distance = 0.178
        self.pose = [0.0, 0.0, 0.0]  # x, y, theta

        
        self.sigma_x = 0.001
        self.sigma_y = 0.001
        self.sigma_z = 0.001
        self.sigma_th = math.radians(0.005)
        self.sigma_v = 0.0005
        self.sigma_w = math.radians(0.005)

        # Noise 
        self.sigma_z_x = 0.005
        self.sigma_z_y = 0.005

        self.v = 0.5
        self.radius = 2
        self.w = self.v / self.radius
        self.dt = 0.1

    def odom_callback(self, msg):
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        _, _, self.yaw = tf_transformations.euler_from_quaternion([x, y, z, w])

        self.pose_x = msg.pose.pose.position.x
        self.pose_y = msg.pose.pose.position.y

    def laser_callback(self, msg):
        pass

    def update(self):
        ksi_x = self.sigma_x * random.gauss(0, 1)
        ksi_y = self.sigma_y * random.gauss(0, 1)
        ksi_th = self.sigma_th * random.gauss(0, 1)
        ksi_v = self.sigma_v * random.gauss(0, 1)
        ksi_w = self.sigma_w * random.gauss(0, 1)

        predicted_pose = self.pose.copy()
        predicted_pose[0] = (predicted_pose[0] + ksi_x) + (self.v + ksi_v) * math.cos(predicted_pose[2] + ksi_th) * self.dt
        predicted_pose[1] = (predicted_pose[1] + ksi_y) + (self.v + ksi_v) * math.sin(predicted_pose[2] + ksi_th) * self.dt
        predicted_pose[2] = (predicted_pose[2] + ksi_th) + (self.w + ksi_w) * self.dt
        self.pose = predicted_pose

        # Measurement (simulated GPS)
        C = np.array([[1, 0, 0], [0, 1, 0]])
        R = np.array([[self.sigma_z_x*2, 0], [0, self.sigma_z_y*2]])
        noise = np.dot(np.sqrt(R), np.random.randn(2, 1)).flatten()
        y = np.dot(C, predicted_pose[:3]) + noise

        # EKF Estimate
        estimated_pose = self.pose.copy()
        estimated_pose[0] += self.v * math.cos(estimated_pose[2]) * self.dt
        estimated_pose[1] += self.v * math.sin(estimated_pose[2]) * self.dt
        estimated_pose[2] += self.w * self.dt

        Q = np.array([[self.sigma_x**2, 0, 0],
                      [0, self.sigma_y**2, 0],
                      [0, 0, self.sigma_th**2]])

        M = np.array([[self.sigma_v**2, 0],
                      [0, self.sigma_w**2]])

        F = np.array([[1, 0, -self.v * math.sin(estimated_pose[2]) * self.dt],
                      [0, 1, self.v * math.cos(estimated_pose[2]) * self.dt],
                      [0, 0, 1]])

        G = np.array([[math.cos(estimated_pose[2]) * self.dt, 0],
                      [math.sin(estimated_pose[2]) * self.dt, 0],
                      [0, self.dt]])

        H = C
        z = np.dot(H, estimated_pose)

        P = Q
        K = np.dot(P, np.dot(H.T, np.linalg.pinv(np.dot(H, np.dot(P, H.T)) + R)))
        estimated_pose = estimated_pose + np.dot(K, (y - z))
        P = np.dot((np.eye(Q.shape[0]) - np.dot(K, H)), P)

        self.pose = estimated_pose
        self.publish_pose()

    def publish_pose(self):
        msg = Pose2D()
        msg.x = self.pose[0]
        msg.y = self.pose[1]
        msg.theta = self.pose[2]
        self.pose_publisher.publish(msg)
        self.get_logger().info(f'Publishing pose -> x: {msg.x:.2f}, y: {msg.y:.2f}, theta: {math.degrees(msg.theta):.2f}°')

    def __del__(self):
        self.get_logger().info('SAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIIIII')


def main(args=None):
    rclpy.init(args=args)
    node = kalmannode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()