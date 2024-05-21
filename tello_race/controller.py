import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32, Float32MultiArray
import cv2
from tello_msgs.srv import TelloAction
from geometry_msgs.msg import Twist
import time


class Timer:
    def __init__(self, max_ticks):
        self.max_ticks = max_ticks
        self.ready = False
        self.curr_ticks = max_ticks

    def reset(self):
        self.curr_ticks = self.max_ticks

    def tick(self):
        if self.curr_ticks > 0:
            self.curr_ticks -= 1
            self.ready = False
        else:
            self.ready = True


class TelloController(Node):
    def __init__(self):
        super().__init__('tello_controller')
        self.get_logger().info("Controller node initialized")
        self.go_through = 0
        self.fps = 15

        # Subscriptions
        self.frame_coord_sub = self.create_subscription(
            Float32MultiArray,
            '/frame_coord',
            self.frame_coord_cb,
            rclpy.qos.qos_profile_sensor_data
        )

        self.fps_sub = self.create_subscription(
            Float32,
            '/fps',
            self.fps_cb,
            rclpy.qos.qos_profile_sensor_data
        )

        # Publisher
        self.tello_vel_pub = self.create_publisher(Twist, '/drone1/cmd_vel', 10)

        # Service Client
        self.tello_client = self.create_client(TelloAction, '/drone1/tello_action')

        # Wait for TelloAction service to become available
        while not self.tello_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for TelloAction service')

        # Takeoff Command
        self.tello_req = TelloAction.Request()
        self.tello_req.cmd = 'takeoff'
        takeoff_future = self.tello_client.call_async(self.tello_req)
        rclpy.spin_until_future_complete(self, takeoff_future)

        self.get_logger().info('Tello took off!')

        # Controller parameters and variables
        self.poi_thresh = 0.15
        self.centralized_frame = False
        self.mean_tgt = [0, 0]
        self.convergence_timeout = 10
        self.search_timeout = 10
        self.search_state = 'START_SEARCH'  # NO_SEARCH, SEARCH_RIGHT, SEARCH_LEFT

    def fps_cb(self, data):
        self.fps = data.data

    def frame_coord_cb(self, data):
        self.get_logger().info(f'Frame data: {data.data}, FPS: {self.fps}')
        self.speed = 0.12 * ((self.fps + 1) / (15 + 1))  # 15 FPS 

        tgt_x, tgt_y, detected_stop, conf_stop = data.data

        self.has_poi = tgt_x > -1000 or tgt_y > -1000
        self.stop_sign = detected_stop == 1.0

        if self.stop_sign and conf_stop > 70000:
            self.stop_sign_close = True
            twist_msg = self.control()
            if twist_msg is not None:
                self.tello_vel_pub.publish(twist_msg)
        else:
            self.stop_sign_close = False

        if self.has_poi:
            self.target = (tgt_x, tgt_y)
            self.mean_tgt[0] = (self.mean_tgt[0] * 8 + tgt_x) / 9
            self.mean_tgt[1] = (self.mean_tgt[1] * 8 + tgt_y) / 9

            # Test point convergence
            if abs(tgt_x - self.mean_tgt[0]) > 0.10 or abs(tgt_y - self.mean_tgt[1]) > 0.10:
                self.get_logger().info('No convergence yet')
                twist_msg = Twist()
                twist_msg.linear.x = self.speed * 0.5 * self.convergence_timeout / 10
                if self.convergence_timeout >= 0:
                    self.convergence_timeout -= 1
                self.tello_vel_pub.publish(twist_msg)
                return
            else:
                self.convergence_timeout = 10

        twist_msg = self.control()
        if twist_msg is not None:
            self.tello_vel_pub.publish(twist_msg)

    def control(self):
        twist_msg = Twist()

        if self.stop_sign_close:
            self.get_logger().info('Landing Tello')
            self.tello_req.cmd = 'land'
            land_future = self.tello_client.call_async(self.tello_req)
            rclpy.spin_until_future_complete(self, land_future)
            self.get_logger().info('Tello Landed')
            return None

        if not self.has_poi:
            if self.centralized_frame and self.centralized_frame_timeout > 0:
                twist_msg.linear.x = self.speed
                self.centralized_frame_timeout -= 1
                self.get_logger().info(f'Passing frame {self.centralized_frame_timeout}')
                return twist_msg

            if self.search_state == 'START_SEARCH':
                self.search_state = 'SEARCH_LEFT'
                self.base_search_timeout = 15
                self.search_timeout = self.base_search_timeout

            if self.search_state == 'SEARCH_LEFT':
                twist_msg.angular.z = self.speed * 2.5
                self.search_timeout -= 1

                if self.search_timeout <= 0:
                    self.search_state = 'SEARCH_RIGHT'
                    self.base_search_timeout *= 2.5
                    self.search_timeout = self.base_search_timeout

                self.get_logger().info('Searching left')
                return twist_msg

            elif self.search_state == 'SEARCH_RIGHT':
                twist_msg.angular.z = -self.speed * 2
                self.search_timeout -= 1

                if self.search_timeout <= 0:
                    self.search_state = 'SEARCH_LEFT'
                    self.base_search_timeout *= 2
                    self.search_timeout = self.base_search_timeout

                    if self.base_search_timeout > 200:
                        self.search_state = 'STOP_SEARCH'

                self.get_logger().info('Searching right')
                return twist_msg

            elif self.search_state == 'STOP_SEARCH':
                self.get_logger().info('Did not find!')
                pass

        else:
            self.search_state = 'START_SEARCH'  # Reset search if point is visible
            tgt_x, tgt_y = self.target
            okX = okY = False

            # Proportional controller
            speed_z = 0.03 + self.speed * abs(tgt_y)
            speed_ang = 0.03 + self.speed * abs(tgt_x)

            # Increase and decrease height
            if tgt_y > self.poi_thresh:
                self.get_logger().info('Go down')
                twist_msg.linear.z = -speed_z
                twist_msg.linear.x = speed_z

            elif tgt_y < -self.poi_thresh:
                self.get_logger().info('Go up')
                twist_msg.linear.z = speed_z
                twist_msg.linear.x = speed_z

            else:
                okX = True

            # Spin horizontally (z axis) to adjust yaw
            if tgt_x > self.poi_thresh:
                self.get_logger().info('Go right')
                twist_msg.angular.z = -speed_ang
                twist_msg.linear.x = 0.0

            elif tgt_x < -self.poi_thresh:
                self.get_logger().info('Go left')
                twist_msg.angular.z = speed_ang
                twist_msg.linear.x = 0.0

            else:
                okY = True

            if okX and okY:
                self.get_logger().info('OK')
                twist_msg.linear.x = self.speed
                self.centralized_frame = True
                self.centralized_frame_timeout = 10
            else:
                self.centralized_frame = False

        return twist_msg


def main(args=None):
    rclpy.init(args=args)
    tello_controller = TelloController()
    rclpy.spin(tello_controller)
    tello_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
