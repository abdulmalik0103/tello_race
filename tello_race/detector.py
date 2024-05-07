import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32MultiArray, Float32

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter

import tensorflow as tf

from tensorflow import preprocess_input, decode_predictions

from mrcnn.config import Config
from mrcnn import model as modellib, utils

LOWER_BROWN = np.array([74, 41, 48])
UPPER_BROWN = np.array([128, 204, 206])

LOWER_GREEN = np.array([40, 40, 40])
UPPER_GREEN = np.array([70, 255, 255])

LOWER_RED = np.array([0, 80, 50])
UPPER_RED = np.array([50, 255, 255])

class TelloSubscriber(Node):

    def __init__(self):
        super().__init__('tello_image_receiver')

        # Load the pre-trained model
        self.model = modellib.MaskRCNN(mode="inference", config=Config, model_dir="/Home/ws_drone/src/tello_race/tello_race/mask_rcnn")
        self.model.load_weights('mask_rcnn_weights.h5', by_name=True)

        self.subscription = self.create_subscription(
            Image,
            '/drone1/image_raw',
            self.image_callback,
            rclpy.qos.qos_profile_sensor_data
        )

        self.frame_coord_pub = self.create_publisher(Float32MultiArray, '/frame_coord', 10)
        self.fps_pub = self.create_publisher(Float32, '/fps', 10)
        self.last_fps_time = time.time()
        self.bridge = CvBridge()

        self.get_logger().info("Image subscriber node initialized with Mask R-CNN")

        self.fps_pub = self.create_publisher(Float32, '/fps', 10)
        self.last_fps_time = time.time()


    def init_kalman(self,):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)

        # position_x, velocity_x, position_y, velocity_y)
        self.kf.x = np.array([[400.], [0.], [400.], [0.]])

        self.kf.P *= 1000.
        self.kf.H = np.array([[1., 0., 0., 0.],
                 [0., 0., 1., 0.]])
        self.kf.R *= 1000  # some reasonable value
        self.kf.Q *= 0.01
        dt = 1  # Time step
        self.kf.F = np.array([[1., dt, 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 1., dt],
                 [0., 0., 0., 1.]])
        return

    def image_callback(self, msg):
        # Convert ROS Image message to CV2 format
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_frame(cv_image)
        
        # Compute and publish the frame rate
        now = time.time()
        fps = 1 / (now - self.last_fps_time)
        self.last_fps_time = now
        fps_msg = Float32()
        fps_msg.data = fps
        self.fps_pub.publish(fps_msg)

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def yolo_search_stop_sign(self, img):
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == self.stop_sign_class_id:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    self.stop_sign_conf = float(w * h)
                    self.detected_stop = 1.0
                    print(f'area: {self.stop_sign_conf}')

                    return center_x, center_y

                    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # cv2.putText(img, "Stop Sign", (x, y - 5),
                    #              cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)        

    def search_stop_sign(self, image):

        cloned_image = image.copy()
        cloned_image = cv2.resize(cloned_image, (224, 224))
        input_image = preprocess_input(cloned_image) 
        predictions = self.model.predict(np.expand_dims(input_image, axis=0))

        decoded_predictions = decode_predictions(predictions, top=5)[0]

        for i, (class_id, class_label, confidence) in enumerate(decoded_predictions):
            if 'street_sign' in class_label.lower():  
                print(f"Detected class: {class_label}, Confidence: {confidence}")
                self.detected_stop = 1.0
                self.stop_sign_conf = confidence

    def process_frame(self, image):

        self.detected_stop = 0.0
        self.stop_sign_conf = 0.0
        self.final_average_point = None
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask_brown = cv2.inRange(hsv, LOWER_BROWN, UPPER_BROWN)
        mask_brown = cv2.erode(mask_brown, self.kernel, iterations=1)
        mask_brown = cv2.dilate(mask_brown, self.kernel, iterations=1)

        mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        mask_green = cv2.erode(mask_green, self.kernel, iterations=1)
        mask_green = cv2.dilate(mask_green, self.kernel, iterations=1)

        mask_red = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
        mask_red = cv2.erode(mask_red, self.kernel, iterations=1)
        mask_red = cv2.dilate(mask_red, self.kernel, iterations=1)

        mask_combined = cv2.bitwise_or(mask_brown, mask_green)
        mask = cv2.bitwise_or(mask_combined, mask_red)


        contours_green, hierarchy_green = cv2.findContours(mask_brown,
                                                cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        contours_brown, hierarchy_brown = cv2.findContours(mask_green,
                                                cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        contours_red, hierarchy_red = cv2.findContours(mask_red,
                                                cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        fin_contour_green = None
        fin_contour_red = None
        fin_contour_brown = None
        if len(contours_green) > 0:
            fin_contour_green, _ = self.get_biggest_inner_contour(contours_green,
                                                                     hierarchy_green, image)
        if len(contours_brown) > 0:                                                           
            fin_contour_brown, _ = self.get_biggest_inner_contour(contours_brown,
                                                                     hierarchy_brown, image)
        if len(contours_red) > 0:  
            fin_contour_red, _ = self.get_biggest_inner_contour(contours_red,
                                                                     hierarchy_red, image)

        final_contours = [fin_contour_green, fin_contour_brown, fin_contour_red]
        valid_contours = [contour for contour in final_contours if contour is not None]
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            self.final_average_point = (center_x, center_y)
            # self.get_logger().info(f'before kalman: {self.final_average_point}')                                
            
            self.kf.predict()
            self.kf.update(self.final_average_point)

            # self.get_logger().info(f'after kalman: {self.final_average_point}')
            cv2.circle(image, self.final_average_point, 5, (255, 127, 127), -1)

        has_point = False

        if self.final_average_point:

            self.final_point_list_filt.append((self.final_average_point[0], self.final_average_point[1]))

            if len(self.final_point_list_filt) > self.filt_size:
                self.final_point_list_filt.pop(0)

            xp = np.mean([p[0] for p in self.final_point_list_filt])
            yp = np.mean([p[1] for p in self.final_point_list_filt])
            has_point = True

            self.stop_sign_conf = cv2.contourArea(largest_contour)

        else: # If no frame was detected, run yolo
            yolo_point = self.yolo_search_stop_sign(image)
            if yolo_point is not None:
                xp, yp = yolo_point
                has_point = True

        if not has_point:
            array = Float32MultiArray(data=[-1000,
                                            -1000,
                                            self.detected_stop ,
                                            self.stop_sign_conf])
        else:
            cv2.circle(image, (int(xp), int(yp)), 5, (255, 255, 0), -1)

            array = Float32MultiArray(data=[(xp / image.shape[1]) * 2 - 1,
                                            (yp / image.shape[0]) * 2 - 1,
                                            self.detected_stop,
                                            self.stop_sign_conf])

        cv2.imshow("res", mask)
        cv2.imshow("Tello Image", image)
        cv2.waitKey(1)

        self.frame_coord_pub.publish(array)

    def get_biggest_inner_contour(self, contours, hierarchy, image):

        cv2.drawContours(image, contours, -1, (255, 100, 100), 2)
        inner_contours = []
        for i, contour in enumerate(contours):

            hierarchy_info = hierarchy[0][i]
    
            if hierarchy_info[3] != -1:  # If the contour has a parent
                parent_index = hierarchy_info[3]
                parent_hierarchy_info = hierarchy[0][parent_index]
                
                has_only_one_child = parent_hierarchy_info[2] == i
                if has_only_one_child:
                    inner_contours.append(contour)
                    cv2.drawContours(image, [contour], -1, (0, 255, 255), 2)     

        areas = [cv2.contourArea(cnt) for cnt in inner_contours]
        output_coords = None
        output_contour = None
        for i, contour in enumerate(inner_contours):
            if cv2.contourArea(contour) > 0:
                M = cv2.moments(contour)
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])
                average_point = (center_x, center_y)
                
                if i == np.argmax(areas):
                    output_contour = contour
                    output_coords = average_point
                    cv2.circle(image, average_point, 5, (255, 0, 0), -1)
                else:
                    cv2.circle(image, average_point, 5, (0, 0, 255), -1)

        return output_contour, output_coords

def main(args=None):
    rclpy.init(args=args)

    tello_sub = TelloSubscriber()

    rclpy.spin(tello_sub)
    tello_sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()