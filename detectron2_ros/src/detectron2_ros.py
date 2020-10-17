#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import threading
import time
ros_path="/opt/ros/kinetic/lib/python2.7/dist-packages"
if '/home/zss/EXTEND/Workspace/voxbloxpp_ws/devel/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/home/zss/EXTEND/Workspace/voxbloxpp_ws/devel/lib/python2.7/dist-packages')
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2 as cv
sys.path.append(ros_path) # 解决ros中的cv2与conda中cv2冲突
import numpy as np
import rospy
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from cv_bridge import CvBridge, CvBridgeError
# import some common detectron2 utilities
sys.path.append('/home/zss/EXTEND/Workspace/voxbloxpp_ws/devel/lib/python2.7/dist-packages')
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2_ros_msg.msg import Result # 把voxbloxpp_ws/devel/lib/python2.7/dist-packages下的的detectron2_ros改名为detectron2_ros_msg
from sensor_msgs.msg import Image, RegionOfInterest


class Detectron2node(object):
    def __init__(self):
        setup_logger()

        self._bridge = CvBridge()
        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._publish_rate = rospy.get_param('~publish_rate', 100)

        self.cfg = get_cfg()
        self.cfg.merge_from_file(rospy.get_param('~model'))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = rospy.get_param(
            '~detection_threshold')  # set threshold for this model
        self.cfg.MODEL.WEIGHTS = "/home/zss/Projects/detectron2/demo/model_final_f10217.pkl" # R50-FPN
        #self.cfg.MODEL.WEIGHTS = "/home/zss/Projects/detectron2/demo/model_final_f6e8b1.pkl" # R101-FPN
        self.predictor = DefaultPredictor(self.cfg)
        self._class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).get("thing_classes", None)

        self._visualization = rospy.get_param('~visualization', True)
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        self._sub = rospy.Subscriber(rospy.get_param('~input'), Image, self._image_callback, queue_size=1)
        
        if self._visualization:
            self._vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)

    def run(self):

        rate = rospy.Rate(self._publish_rate)
        i = 0
        sum_time = 0
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                img_msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if img_msg is not None:
                start = time.time()                
                np_image = self.convert_to_cv_image(img_msg)

                outputs = self.predictor(np_image)
                result = outputs["instances"].to("cpu")
                result_msg = self.getResult(result)

                self._result_pub.publish(result_msg)
                end = time.time()
                once = end - start
                sum_time = sum_time + once
                i = i + 1
                #print('time:',once,'s')
                average_time=sum_time/i
                print('average time:',average_time)
                # Visualize results
                if self._visualization:
                    v = Visualizer(np_image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
                    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    img = v.get_image()[:, :, ::-1]

                    image_msg = self._bridge.cv2_to_imgmsg(img)
                    self._vis_pub.publish(image_msg)

            rate.sleep()
        
    def getResult(self, predictions):

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
        else:
            return

        result_msg = Result()
        result_msg.header = self._header
        result_msg.class_ids = predictions.pred_classes if predictions.has("pred_classes") else None
        result_msg.class_names = np.array(self._class_names)[result_msg.class_ids.numpy()]
        result_msg.scores = predictions.scores if predictions.has("scores") else None

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]]=255
            mask = self._bridge.cv2_to_imgmsg(mask)
            mask.encoding = "mono8" # 参考mask_rcnn_node line164
            result_msg.masks.append(mask)

            box = RegionOfInterest()
            box.x_offset = np.uint32(x1)
            box.y_offset = np.uint32(y1)
            box.height = np.uint32(y2 - y1)
            box.width = np.uint32(x2 - x1)
            result_msg.boxes.append(box)

        return result_msg

    def convert_to_cv_image(self, image_msg):

        if image_msg is None:
            return None

        self._width = image_msg.width
        self._height = image_msg.height
        channels = int(len(image_msg.data) / (self._width * self._height))

        encoding = None
        if image_msg.encoding.lower() in ['rgb8', 'bgr8']:
            encoding = np.uint8
        elif image_msg.encoding.lower() == 'mono8':
            encoding = np.uint8
        elif image_msg.encoding.lower() == '32fc1':
            encoding = np.float32
            channels = 1

        cv_img = np.ndarray(shape=(image_msg.height, image_msg.width, channels),
                            dtype=encoding, buffer=image_msg.data)

        if image_msg.encoding.lower() == 'mono8':
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)
        else:
            cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2BGR)

        return cv_img

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._header = msg.header
            self._msg_lock.release()



def main(argv):
    rospy.init_node('detectron2_ros')
    node = Detectron2node()
    node.run()

if __name__ == '__main__':
    main(sys.argv)
