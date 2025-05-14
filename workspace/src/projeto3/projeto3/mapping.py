import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
import tf_transformations
from bresenham import bresenham
import numpy as np
import math
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

class MapeamentoNode(Node):
    def __init__(self):
        super().__init__('mapeamento_node')

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        self.pose = None
        self.map_res = 0.05 
        self.map_size = 400 
        self.map = np.full((self.map_size, self.map_size), -1, dtype=np.int8)  
        self.origin_x = -10.0
        self.origin_y = -10.0

        # Broadcaster para transformar de "map" para "odom"
        self.broadcaster = StaticTransformBroadcaster(self)
        self.publicar_static_transform()

    def publicar_static_transform(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0

        q = tf_transformations.quaternion_from_euler(0, 0, 0)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.broadcaster.sendTransform(t)

    def odom_callback(self, msg):
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        _, _, yaw = tf_transformations.euler_from_quaternion([x, y, z, w])

        pos_x = msg.pose.pose.position.x
        pos_y = msg.pose.pose.position.y
        self.pose = (pos_x, pos_y, yaw)
        
    def laser_callback(self, msg):
        if self.pose is None:
            return

        robot_x, robot_y, robot_theta = self.pose
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment

        for i, r in enumerate(msg.ranges):
            if math.isinf(r) or math.isnan(r):
                continue

            angle = angle_min + i * angle_increment + robot_theta
            hit_x = robot_x + r * math.cos(angle)
            hit_y = robot_y + r * math.sin(angle)

            x0 = int((robot_x - self.origin_x) / self.map_res)
            y0 = int((robot_y - self.origin_y) / self.map_res)
            x1 = int((hit_x - self.origin_x) / self.map_res)
            y1 = int((hit_y - self.origin_y) / self.map_res)

            l = list(bresenham(x0, y0, x1, y1))
            for x, y in l[:-1]:  # todas menos a última célula (evita marcar o obstáculo como livre)
                if 0 <= x < self.map_size and 0 <= y < self.map_size:
                    self.map[y, x] = 0  # livre

            if 0 <= x1 < self.map_size and 0 <= y1 < self.map_size:
                self.map[y1, x1] = 100  # obstáculo

        self.publicar_mapa()

    def publicar_mapa(self):
        grid = OccupancyGrid()
        grid.header.frame_id = "map"
        grid.header.stamp = self.get_clock().now().to_msg()
        grid.info.resolution = self.map_res
        grid.info.width = self.map_size
        grid.info.height = self.map_size
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        grid.data = self.map.flatten().tolist()
        self.map_pub.publish(grid)

def main(args=None):
    rclpy.init(args=args)
    node = MapeamentoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
