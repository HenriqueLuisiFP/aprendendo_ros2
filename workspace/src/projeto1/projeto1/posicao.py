import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import Header
import time

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

        # Timer to publish messages every 0.5 seconds
        self.timer = self.create_timer(0.5, self.talker_callback_joint)

        # Initialize joint positions
        self.jointL = 0.0
        self.jointR = 0.0

    def run(self):
        rclpy.spin(self)

    def listener_callback_joint(self, msg):
        self.jointL = msg.position[0]
        self.jointR = msg.position[1]

    def listener_callback_laser(self, msg):
        self.laser = msg.ranges[80]

    def talker_callback_joint(self):
        msg = JointState()
        
        # 🔥 FIX: Set Header (Mandatory for Some Subscribers)
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"  # Change if needed

        msg.position = [self.jointL, self.jointR]
        

        self.publisher.publish(msg)
        self.get_logger().info(f'Published JointState: {msg.position}')

    def __del__(self):
        self.get_logger().info('Finalizando o nó! Tchau, tchau...')

# Main function
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


