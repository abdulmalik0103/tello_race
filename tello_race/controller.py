import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray
from geometry_msgs.msg import Twist
from tello_msgs.srv import TelloAction
import time

class TelloController(Node):
    def __init__(self):
        super().__init__('tello_controller')
        self.get_logger().info("Controller node initialized")

        self.frame_coord_sub = self.create_subscription(
            Float32MultiArray, '/frame_coord', self.frame_coord_callback, 10)
        self.fps_sub = self.create_subscription(
            Float32, '/fps', self.fps_callback, 10)

        self.tello_vel_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)
        self.tello_client = self.create_client(TelloAction, '/drone1/tello_action')

        self.fps = 15
        self.poi_threshold = 0.15
        self.convergence_timeout = 10
        self.search_timeout = 10
        self.search_state = 'START_SEARCH'
        self.centralized_frame = False

        self.initiate_takeoff()

    def initiate_takeoff(self):
        while not self.tello_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for TelloAction service...')
        
        self.get_logger().info('Sending takeoff command...')
        takeoff_req = TelloAction.Request()
        takeoff_req.cmd = 'takeoff'
        self.tello_client.call_async(takeoff_req)

    def fps_callback(self, msg):
        self.fps = msg.data

    def frame_coord_callback(self, msg):
        data = msg.data
        tgt_x, tgt_y, detected_stop, conf_stop = data
        self.process_frame_data(tgt_x, tgt_y, detected_stop, conf_stop)

    def process_frame_data(self, tgt_x, tgt_y, detected_stop, conf_stop):
        if detected_stop == 1.0 and conf_stop > 70000:
            self.execute_landing()
            return

        if tgt_x != -1000 and tgt_y != -1000:
            self.control_drone(tgt_x, tgt_y)

    def control_drone(self, tgt_x, tgt_y):
        twist_msg = Twist()

        # Proportional control calculations
        twist_msg.linear.z = self.calculate_axis_speed(tgt_y, 'z')
        twist_msg.angular.z = self.calculate_axis_speed(tgt_x, 'yaw')
        twist_msg.linear.x = self.speed_from_fps()

        if self.is_target_centralized(tgt_x, tgt_y):
            self.get_logger().info('Target centralized. Proceeding...')
            self.centralized_frame = True
            self.tello_vel_pub.publish(twist_msg)
        else:
            self.get_logger().info('Adjusting to centralize target...')
            self.centralized_frame = False
            self.tello_vel_pub.publish(twist_msg)

    def calculate_axis_speed(self, target_offset, axis):
        speed = 0.03 + 0.12 * abs(target_offset)  # Example proportional control
        return speed if target_offset > self.poi_threshold else -speed if target_offset < -self.poi_threshold else 0

    def speed_from_fps(self):
        # Adjust speed based on FPS for smoother control
        return 0.12 * ((self.fps + 1) / (15 + 1))  # Normalize speed based on FPS

    def is_target_centralized(self, tgt_x, tgt_y):
        return abs(tgt_x) < self.poi_threshold and abs(tgt_y) < self.poi_threshold

    def execute_landing(self):
        self.get_logger().info('Landing Tello...')
        land_req = TelloAction.Request()
        land_req.cmd = 'land'
        self.tello_client.call_async(land_req)

def main(args=None):
    rclpy.init(args=args)
    tello_controller = TelloController()
    rclpy.spin(tello_controller)
    tello_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
