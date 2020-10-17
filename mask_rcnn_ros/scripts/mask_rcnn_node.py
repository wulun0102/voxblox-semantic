#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import threading
import numpy as np
import cv2
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import time
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest

from mask_rcnn_ros import coco
from mask_rcnn_ros import utils
from mask_rcnn_ros import model as modellib
from mask_rcnn_ros import visualize
from mask_rcnn_ros.msg import Result
#import tensorflow as tf
from tensorflow import ConfigProto
from tensorflow import InteractiveSession


config = ConfigProto()  
config.gpu_options.allow_growth = True # 为限制GPU资源占用，动态申请显存
session = InteractiveSession(config=config)

# 本地权重文件的路径
ROS_HOME = os.environ.get('ROS_HOME', os.path.join(os.environ['HOME'], '.ros'))
# COCO_MODEL_PATH = os.path.join(ROS_HOME, 'mask_rcnn_coco.h5')  # use local mask_rcnn_coco.h5
COCO_MODEL_PATH = os.path.join('/home/zss/EXTEND/dataset', 'mask_rcnn_coco.h5')
RGB_TOPIC = '/camera/rgb/image_raw'

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class,查找对象的索引位置
#  use: CLASS_NAMES.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class InferenceConfig(coco.CocoConfig): # 继承CocoConfig类，用于进行推断用的配置
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #IMAGE_MIN_DIM = 128
    #IMAGE_MAX_DIM = 128


class MaskRCNNNode(object):
    def __init__(self):
        self._cv_bridge = CvBridge()

        config = InferenceConfig()
        config.display()

        # Get input RGB topic.
        self._rgb_input_topic = rospy.get_param('~input', RGB_TOPIC)

        self._visualization = rospy.get_param('~visualization', True)

        # Create model object in inference mode.
        self._model = modellib.MaskRCNN(mode="inference", model_dir="",
                                        config=config)
        # Load weights trained on MS-COCO
        # model_path = rospy.get_param('~model_path', COCO_MODEL_PATH)
        model_path = rospy.get_param('~model_path', '/home/zss/EXTEND/DeepLearningModel/mask_rcnn_coco.h5')
        # Download COCO trained weights from Releases if needed
        # if model_path == COCO_MODEL_PATH and not os.path.exists(COCO_MODEL_PATH):
        #   utils.download_trained_weights(COCO_MODEL_PATH)

        self._model.load_weights(model_path, by_name=True)

        self._class_names = rospy.get_param('~class_names', CLASS_NAMES)

        self._last_msg = None
        self._msg_lock = threading.Lock() # 定义了一个线程锁

        self._class_colors = visualize.random_colors(len(CLASS_NAMES))

        self._publish_rate = rospy.get_param('~publish_rate', 100)

    def run(self):
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)# 定义一个发布者，发布result话题
        vis_pub = rospy.Publisher('~visualization', Image, queue_size=1)

        rospy.Subscriber(self._rgb_input_topic, Image,
                         self._image_callback, queue_size=1)# 定义一个订阅者，接收RGB_TOPIC

        rate = rospy.Rate(self._publish_rate)
        i = 0
        sum_time = 0
        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):# 返回值指定线程是否获得了锁
                msg = self._last_msg # 从RGB_TOPIC传来的msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                start = time.time() 
                np_image = self._cv_bridge.imgmsg_to_cv2(msg, 'bgr8')# msg代表图像消息

                # Run detection,对图像列表进行检测,返回一个包含字典的列表，一个字典对应一个图像的处理结果
                results = self._model.detect([np_image], verbose=0)
                result = results[0]
                result_msg = self._build_result_msg(msg, result)
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
                    cv_result = self._visualize_plt(result, np_image)
                    image_msg = self._cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                    vis_pub.publish(image_msg)

            rate.sleep()

    def _build_result_msg(self, msg, result):# 从推理结果result中打包生成一个result_msg
        result_msg = Result()
        result_msg.header = msg.header # 把图像的时间戳赋给推理msg
        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            box = RegionOfInterest() # 把推理的bounding box赋值给msg
            box.x_offset = np.asscalar(x1)
            box.y_offset = np.asscalar(y1)
            box.height = np.asscalar(y2 - y1)
            box.width = np.asscalar(x2 - x1)
            result_msg.boxes.append(box)

            class_id = result['class_ids'][i]
            result_msg.class_ids.append(class_id)

            class_name = self._class_names[class_id]
            result_msg.class_names.append(class_name)

            score = result['scores'][i]
            result_msg.scores.append(score)

            mask = Image()
            mask.header = msg.header # 看Mask的定义？?
            mask.height = result['masks'].shape[0]
            mask.width = result['masks'].shape[1]
            mask.encoding = "mono8"
            mask.is_bigendian = False
            mask.step = mask.width
            mask.data = (result['masks'][:, :, i] * 255).tobytes()
            result_msg.masks.append(mask)
        return result_msg

    def _visualize(self, result, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        visualize.display_instances(image, result['rois'], result['masks'],
                                    result['class_ids'], CLASS_NAMES,
                                    result['scores'], ax=axes,
                                    class_colors=self._class_colors)
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result

    def _get_fig_ax(self):
        """Return a Matplotlib Axes array to be used in
        all visualizations. Provide a
        central point to control graph sizes.

        Change the default size attribute to control the size
        of rendered images
        """
        fig, ax = plt.subplots(1)
        plt.subplots_adjust(
            left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        return fig, ax

    def _visualize_plt(self, result, image):
        fig, ax = self._get_fig_ax()
        image = visualize.display_instances_plt(
            image,
            result['rois'],
            result['masks'],
            result['class_ids'],
            CLASS_NAMES,
            result['scores'],
            fig=fig,
            ax=ax)

        return image

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()


def main():
    rospy.init_node('mask_rcnn')

    node = MaskRCNNNode()
    node.run()


if __name__ == '__main__':
    main()
