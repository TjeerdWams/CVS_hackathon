#!/usr/bin/env python3

# Python
from functools import partial
import torch

# Imaging
import numpy as np
import cv2 # This has to be imported before CVBridge, otherwise errors occur

# ROS
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


# YOLOv10n inference class
class YoloV10nDetector:
    def __init__(self, model_path, class_names):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.class_names = class_names

    def preprocess(self, image):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_transposed = img_resized.transpose(2, 0, 1)
        img_normalized = img_transposed / 255.0
        img_tensor = torch.from_numpy(img_normalized).float().to(self.device)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def postprocess(self, preds, confidence_threshold=0.5):
        detections = []
        for pred in preds:
            scores = pred[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id] * pred[4]
            if confidence > confidence_threshold:
                x1, y1, x2, y2 = map(int, pred[:4])
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id]
                })
        return detections

    def detect(self, image):
        img_tensor = self.preprocess(image)
        with torch.no_grad():
            preds = self.model(img_tensor)[0]
        detections = self.postprocess(preds)
        return detections



# Feel free to modify this class to your liking and needs
# The ROS node should work out of the box, you don't need to edit that
class SummerSchool():
    def __init__(self, model_path, class_names):
        self.node = rospy.init_node('summer_school', anonymous=False)
        self.image_subscriber = rospy.Subscriber("sensors/front_camera/image_raw", Image, partial(SummerSchool.image_callback, self))
        self.vel_publisher = rospy.Publisher('locomotion/cmd_vel', Twist, queue_size=10)
        self.bridge = CvBridge()
        self.detector = YoloV10nDetector(model_path, class_names)
        rospy.loginfo("Summer school node 2 is ready.")


    # This function is called every time this ROS node receives an image from the camera
    def image_callback(self, image_msg):
        # This converts the image message to an (numpy) array with image data 
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        # The variable named 'image' contains an array with the image data
        # With the default setup it has the following shape (720, 1280, 3)
        # Your code goes here

        detections = self.detector.detect(image)
        dangerous_rock = any(d['class_name'] == 'DANGER_ROCK' and d['confidence'] > 0.4 for d in detections)

        # Run detection
        if dangerous_rock:
            self.twist_msg_publisher(lin_vel=0.0, ang_vel=0.0)


    # This function serves as an example on how to publish twist messages to control the rover
    # You can just call this function or use the code here to implement your own solution
    # Calling this function once results in sending one message on the cmd_vel topic which our motor controller listens 
    def twist_msg_publisher(self, lin_vel, ang_vel):
        # Twist message is documented here: http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Twist.html
        twist_msg = Twist()
        twist_msg.linear.x = lin_vel
        twist_msg.angular.z = ang_vel
        self.vel_publisher.publish(twist_msg)


# Starting the node
if __name__ == "__main__":
    
    model_path = "src/rock_yolo_model.pt"
    class_names = ['DANGER_ROCK', 'SAFE_ROCK'] 
    summer_school = SummerSchool(model_path, class_names)
    # This keeps the node alive until roscore is running
    rospy.spin()

