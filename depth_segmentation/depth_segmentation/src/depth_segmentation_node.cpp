#include <cv_bridge/cv_bridge.h>
#include <dynamic_reconfigure/server.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <detectron2_ros/Result.h> //和下面的result.h完全相同，故下面代码没有改成detectron2_ros包下的result
#ifdef MASKRCNNROS_AVAILABLE
#include <mask_rcnn_ros/Result.h>
#endif

#include "depth_segmentation/depth_segmentation.h"
#include "depth_segmentation/ros_common.h"

struct PointSurfelLabel {
  PCL_ADD_POINT4D;
  PCL_ADD_NORMAL4D;
  PCL_ADD_RGB;
  uint8_t instance_label;
  uint8_t semantic_label;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW// 确保new操作符对齐操作
} EIGEN_ALIGN16;//声明struct同时定义结构变量，在eigen中存在宏EIGEN_ALIGN16，用于内存对齐

POINT_CLOUD_REGISTER_POINT_STRUCT(// 注册点类型宏，以后才能使用自定义类型的点
    PointSurfelLabel,
    (float, x, x)(float, y, y)(float, z, z)(float, normal_x, normal_x)(
        float, normal_y, normal_y)(float, normal_z, normal_z)(float, rgb, rgb)(//rgb为什么是一个float而不是struct？
        uint8_t, instance_label, instance_label)(uint8_t, semantic_label,
                                                 semantic_label))

class DepthSegmentationNode {
 public:
  DepthSegmentationNode() //构造函数中创建了节点句柄、发布者和订阅者，而发布消息的函数在publish_segments()
      : node_handle_("~"),
        image_transport_(node_handle_),
        camera_info_ready_(false),
        depth_camera_(),
        rgb_camera_(),
        params_(),
        camera_tracker_(depth_camera_, rgb_camera_),
        depth_segmenter_(depth_camera_, params_) {
          //从common.h载入两个参数
    node_handle_.param<bool>("semantic_instance_segmentation/enable",      //是否启用语义分割
                             params_.semantic_instance_segmentation.enable,
                             params_.semantic_instance_segmentation.enable);
    node_handle_.param<float>(
        "semantic_instance_segmentation/overlap_threshold",
        params_.semantic_instance_segmentation.overlap_threshold,
        params_.semantic_instance_segmentation.overlap_threshold);
    //前四个订阅关于rgbd的话题参数默认从primesense_topics.yaml载入，后三个由ros_common.h载入
    node_handle_.param<std::string>("depth_image_sub_topic", depth_image_topic_,
                                    depth_segmentation::kDepthImageTopic);
    node_handle_.param<std::string>("rgb_image_sub_topic", rgb_image_topic_,
                                    depth_segmentation::kRgbImageTopic);
    node_handle_.param<std::string>("depth_camera_info_sub_topic",
                                    depth_camera_info_topic_,
                                    depth_segmentation::kDepthCameraInfoTopic);
    node_handle_.param<std::string>("rgb_camera_info_sub_topic",
                                    rgb_camera_info_topic_,
                                    depth_segmentation::kRgbCameraInfoTopic);
    node_handle_.param<std::string>(
        "semantic_instance_segmentation_sub_topic",
        semantic_instance_segmentation_topic_,
        depth_segmentation::kSemanticInstanceSegmentationTopic);// /mask_rcnn/result

    node_handle_.param<std::string>("world_frame", world_frame_,
                                    depth_segmentation::kTfWorldFrame);
    node_handle_.param<std::string>("camera_frame", camera_frame_,
                                    depth_segmentation::kTfDepthCameraFrame);

    //image_transport 管理同步rgbd图像和CameraInfo主题的订阅同步回调？
    depth_image_sub_ = new image_transport::SubscriberFilter(
        image_transport_, depth_image_topic_, 1);
    rgb_image_sub_ = new image_transport::SubscriberFilter(image_transport_,
                                                           rgb_image_topic_, 1);
    depth_info_sub_ = new message_filters::Subscriber<sensor_msgs::CameraInfo>(
        node_handle_, depth_camera_info_topic_, 1);
    rgb_info_sub_ = new message_filters::Subscriber<sensor_msgs::CameraInfo>(
        node_handle_, rgb_camera_info_topic_, 1);

    constexpr int kQueueSize = 30;

#ifndef MASKRCNNROS_AVAILABLE
     if (params_.semantic_instance_segmentation.enable) {
      params_.semantic_instance_segmentation.enable = false;
      ROS_WARN_STREAM(
          "Turning off semantic instance segmentation "
          "as mask_rcnn_ros is disabled.");
    }
#endif

    if (params_.semantic_instance_segmentation.enable) {
#ifdef MASKRCNNROS_AVAILABLE //在cmakelists中已经定义-DMASKRCNNROS_AVAILABLE
//同步RGB图像、深度图像和mask_rcnn分割的result消息
      instance_segmentation_sub_ =
          new message_filters::Subscriber<mask_rcnn_ros::Result>(
              node_handle_, semantic_instance_segmentation_topic_, 1);

      image_segmentation_sync_policy_ =  //使用过滤规则ImageSegmentationSyncPolicy建立同步器
          new message_filters::Synchronizer<ImageSegmentationSyncPolicy>(
              ImageSegmentationSyncPolicy(kQueueSize), *depth_image_sub_,
              *rgb_image_sub_, *instance_segmentation_sub_);

      image_segmentation_sync_policy_->registerCallback(boost::bind(
          &DepthSegmentationNode::imageSegmentationCallback, this, _1, _2, _3));
#endif
    } else {//只需同步RGB图像和深度图像
      image_sync_policy_ = new message_filters::Synchronizer<ImageSyncPolicy>(
          ImageSyncPolicy(kQueueSize), *depth_image_sub_, *rgb_image_sub_);

      image_sync_policy_->registerCallback(
          boost::bind(&DepthSegmentationNode::imageCallback, this, _1, _2));
    }
//同步depth info和rgb info
    camera_info_sync_policy_ =
        new message_filters::Synchronizer<CameraInfoSyncPolicy>(
            CameraInfoSyncPolicy(kQueueSize), *depth_info_sub_, *rgb_info_sub_);

    camera_info_sync_policy_->registerCallback(
        boost::bind(&DepthSegmentationNode::cameraInfoCallback, this, _1, _2));
//pcl2 segment发布者
    point_cloud2_segment_pub_ =
        node_handle_.advertise<sensor_msgs::PointCloud2>("object_segment",
                                                         1000);
    point_cloud2_scene_pub_ =
        node_handle_.advertise<sensor_msgs::PointCloud2>("segmented_scene", 1);

    node_handle_.param<bool>("visualize_segmented_scene",
                             params_.visualize_segmented_scene,
                             params_.visualize_segmented_scene);
  }

 private:
  ros::NodeHandle node_handle_;
  image_transport::ImageTransport image_transport_;//imageTransport类似于ros:：NodeHandle，因为它包含advertise（）和subscribe（）函数，用于创建图像主题的发布和订阅。
  tf::TransformBroadcaster transform_broadcaster_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image,
                                                          sensor_msgs::Image>
      ImageSyncPolicy;

#ifdef MASKRCNNROS_AVAILABLE
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image, mask_rcnn_ros::Result>
      ImageSegmentationSyncPolicy;
#endif

  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>
      CameraInfoSyncPolicy;

  bool camera_info_ready_;
  depth_segmentation::DepthCamera depth_camera_;
  depth_segmentation::RgbCamera rgb_camera_;

  depth_segmentation::Params params_;

 public://作用？
  depth_segmentation::CameraTracker camera_tracker_;
  depth_segmentation::DepthSegmenter depth_segmenter_;

 private:
  std::string rgb_image_topic_;
  std::string rgb_camera_info_topic_;
  std::string depth_image_topic_;
  std::string depth_camera_info_topic_;
  std::string semantic_instance_segmentation_topic_;
  std::string world_frame_;
  std::string camera_frame_;

  image_transport::SubscriberFilter* depth_image_sub_;
  image_transport::SubscriberFilter* rgb_image_sub_;

  message_filters::Subscriber<sensor_msgs::CameraInfo>* depth_info_sub_;
  message_filters::Subscriber<sensor_msgs::CameraInfo>* rgb_info_sub_;

  ros::Publisher point_cloud2_segment_pub_;
  ros::Publisher point_cloud2_scene_pub_;

  message_filters::Synchronizer<ImageSyncPolicy>* image_sync_policy_;

  message_filters::Synchronizer<CameraInfoSyncPolicy>* camera_info_sync_policy_;

#ifdef MASKRCNNROS_AVAILABLE
  message_filters::Subscriber<mask_rcnn_ros::Result>*
      instance_segmentation_sub_;
  message_filters::Synchronizer<ImageSegmentationSyncPolicy>*
      image_segmentation_sync_policy_;
#endif

  void publish_tf(const cv::Mat cv_transform, const ros::Time& timestamp) {
    // Rotate such that the world frame initially aligns with the camera_link
    // frame.设置初始位姿，使世界坐标系与摄相机坐标系最开始对齐
    static const cv::Mat kWorldAlign =
        (cv::Mat_<double>(4, 4) << 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0,
         1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    cv::Mat cv_transform_world_aligned = cv_transform * kWorldAlign;
    //传入的cv_transform是CameraTracker中的world transform,？为什么与初始位姿相乘得到camera_frame_到world_frame_的变换？
    tf::Vector3 translation_tf(cv_transform_world_aligned.at<double>(0, 3),
                               cv_transform_world_aligned.at<double>(1, 3),
                               cv_transform_world_aligned.at<double>(2, 3));

    tf::Matrix3x3 rotation_tf;
    for (size_t i = 0u; i < 3u; ++i) {
      for (size_t j = 0u; j < 3u; ++j) {
        rotation_tf[j][i] = cv_transform_world_aligned.at<double>(j, i);
      }
    }
    tf::Transform transform;
    transform.setOrigin(translation_tf);
    transform.setBasis(rotation_tf);

    transform_broadcaster_.sendTransform(tf::StampedTransform( //transform定义了camera_frame_到world_frame_的变换T_wc
        transform, timestamp, camera_frame_, world_frame_));
  }

  void fillPoint(const cv::Vec3f& point, const cv::Vec3f& normals,
                 const cv::Vec3f& colors, pcl::PointSurfel* point_pcl) {
    point_pcl->x = point[0];
    point_pcl->y = point[1];
    point_pcl->z = point[2];
    point_pcl->normal_x = normals[0];
    point_pcl->normal_y = normals[1];
    point_pcl->normal_z = normals[2];
    point_pcl->r = colors[0];
    point_pcl->g = colors[1];
    point_pcl->b = colors[2];
  }

  void fillPoint(const cv::Vec3f& point, const cv::Vec3f& normals,
                 const cv::Vec3f& colors, const size_t& semantic_label,
                 const size_t& instance_label, PointSurfelLabel* point_pcl) {
    point_pcl->x = point[0];
    point_pcl->y = point[1];
    point_pcl->z = point[2];
    point_pcl->normal_x = normals[0];
    point_pcl->normal_y = normals[1];
    point_pcl->normal_z = normals[2];
    point_pcl->r = colors[0];
    point_pcl->g = colors[1];
    point_pcl->b = colors[2];

    point_pcl->semantic_label = semantic_label;
    point_pcl->instance_label = instance_label;
  }

  void publish_segments(
      const std::vector<depth_segmentation::Segment>& segments,//从哪里传入的segments？
      const std_msgs::Header& header) {
    CHECK_GT(segments.size(), 0u);//如果检测为true，则返回NULL，否则就会返回一个有明确提示信息的字符串指针，并输出该信息，然后是程序宕掉。0u是无符号整型
    // Just for rviz also publish the whole scene, as otherwise only ~10
    // segments are shown:
    // https://github.com/ros-visualization/rviz/issues/689
    sensor_msgs::PointCloud2 pcl2_msg;

    if (params_.semantic_instance_segmentation.enable) {
      pcl::PointCloud<PointSurfelLabel>::Ptr scene_pcl(
          new pcl::PointCloud<PointSurfelLabel>);
      for (depth_segmentation::Segment segment : segments) {
        CHECK_GT(segment.points.size(), 0u);
        pcl::PointCloud<PointSurfelLabel>::Ptr segment_pcl(
            new pcl::PointCloud<PointSurfelLabel>);
        for (std::size_t i = 0u; i < segment.points.size(); ++i) {//遍历segment的所有points，把segment的点的信息赋值给自己定义的点PointSurfelLabel
          PointSurfelLabel point_pcl;
          uint8_t semantic_label = 0u;
          uint8_t instance_label = 0u;
          if (segment.instance_label.size() > 0u) {
            instance_label = *(segment.instance_label.begin());//返回segment的label set的第一个？
            semantic_label = *(segment.semantic_label.begin());
          }
          fillPoint(segment.points[i], segment.normals[i],
                    segment.original_colors[i], semantic_label, instance_label,
                    &point_pcl);

          segment_pcl->push_back(point_pcl);
          scene_pcl->push_back(point_pcl);
        }
        sensor_msgs::PointCloud2 pcl2_msg;
        pcl::toROSMsg(*segment_pcl, pcl2_msg);
        pcl2_msg.header.stamp = header.stamp;
        pcl2_msg.header.frame_id = header.frame_id;
        point_cloud2_segment_pub_.publish(pcl2_msg);//不断发布由一个个Segment转换为rosmsg格式的pcl2_msg
      }
      if (params_.visualize_segmented_scene) {//在rviz上可视化所有segments
        pcl::toROSMsg(*scene_pcl, pcl2_msg);
      }
    } else {
      pcl::PointCloud<pcl::PointSurfel>::Ptr scene_pcl(
          new pcl::PointCloud<pcl::PointSurfel>);
      for (depth_segmentation::Segment segment : segments) {//不启用语义分割时，segment就不包含label
        CHECK_GT(segment.points.size(), 0u);
        pcl::PointCloud<pcl::PointSurfel>::Ptr segment_pcl(
            new pcl::PointCloud<pcl::PointSurfel>);
        for (std::size_t i = 0u; i < segment.points.size(); ++i) {
          pcl::PointSurfel point_pcl;

          fillPoint(segment.points[i], segment.normals[i],   
                    segment.original_colors[i], &point_pcl);

          segment_pcl->push_back(point_pcl);
          scene_pcl->push_back(point_pcl);
        }
        sensor_msgs::PointCloud2 pcl2_msg;
        pcl::toROSMsg(*segment_pcl, pcl2_msg);
        pcl2_msg.header.stamp = header.stamp;
        pcl2_msg.header.frame_id = header.frame_id;
        point_cloud2_segment_pub_.publish(pcl2_msg);
      }
      if (params_.visualize_segmented_scene) {
        pcl::toROSMsg(*scene_pcl, pcl2_msg);
      }
    }

    if (params_.visualize_segmented_scene) {
      pcl2_msg.header.stamp = header.stamp;
      pcl2_msg.header.frame_id = header.frame_id;
      point_cloud2_scene_pub_.publish(pcl2_msg);
    }
  }

#ifdef MASKRCNNROS_AVAILABLE 
  void semanticInstanceSegmentationFromRosMsg(//把result msg转换为SemanticInstanceSegmentation里的masks和labels
      const mask_rcnn_ros::Result::ConstPtr& segmentation_msg,
      depth_segmentation::SemanticInstanceSegmentation*
          semantic_instance_segmentation) {
    semantic_instance_segmentation->masks.reserve(//根据指定的元素数量，reserve函数为vector预先分配需要的内存
        segmentation_msg->masks.size());
    semantic_instance_segmentation->labels.reserve(
        segmentation_msg->masks.size());
    for (size_t i = 0u; i < segmentation_msg->masks.size(); ++i) {
      cv_bridge::CvImagePtr cv_mask_image;
      cv_mask_image = cv_bridge::toCvCopy(segmentation_msg->masks[i],//拷贝了result里的sensor_msgs/Image的内容，转换为一个opencv图像
                                          sensor_msgs::image_encodings::MONO8);//CV_8UC1， 灰度图像
      semantic_instance_segmentation->masks.push_back(
          cv_mask_image->image.clone());  
      semantic_instance_segmentation->labels.push_back(
          segmentation_msg->class_ids[i]);
    }
  }
#endif

  void preprocess(const sensor_msgs::Image::ConstPtr& depth_msg,
                  const sensor_msgs::Image::ConstPtr& rgb_msg,
                  cv::Mat* rescaled_depth, cv::Mat* dilated_rescaled_depth,
                  cv_bridge::CvImagePtr cv_rgb_image,
                  cv_bridge::CvImagePtr cv_depth_image, cv::Mat* bw_image,
                  cv::Mat* mask) {
    CHECK_NOTNULL(rescaled_depth);
    CHECK_NOTNULL(dilated_rescaled_depth);
    CHECK(cv_rgb_image);
    CHECK(cv_depth_image);
    CHECK_NOTNULL(bw_image);
    CHECK_NOTNULL(mask);
    //--- step1.保证深度图都是CV_32FC1编码
    if (depth_msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
      cv_depth_image = cv_bridge::toCvCopy(
          depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
      *rescaled_depth = cv::Mat::zeros(cv_depth_image->image.size(), CV_32FC1);
      cv::rgbd::rescaleDepth(cv_depth_image->image, CV_32FC1, *rescaled_depth);//输入的图像是CV_16UC1编码，转换为CV_32FC1
    } else if (depth_msg->encoding ==
               sensor_msgs::image_encodings::TYPE_32FC1) {
      cv_depth_image = cv_bridge::toCvCopy(
          depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
      *rescaled_depth = cv_depth_image->image;
    } else {
      LOG(FATAL) << "Unknown depth image encoding.";
    }

    constexpr double kZeroValue = 0.0;
    cv::Mat nan_mask = *rescaled_depth != *rescaled_depth;//*rescaled_depth != *rescaled_depth有什么用？
    rescaled_depth->setTo(kZeroValue, nan_mask);//nan_mask和rescaled_depth同尺寸，把nan_mask的不为0的元素位置对应在rescaled_depth的元素设置为0
    //---step2.深度图膨胀，为了去噪？
    if (params_.dilate_depth_image) {
      cv::Mat element = cv::getStructuringElement(//得到结构元素，第一个参数是形状，第二个是大小
          cv::MORPH_RECT, cv::Size(2u * params_.dilation_size + 1u,
                                   2u * params_.dilation_size + 1u));
      cv::morphologyEx(*rescaled_depth, *dilated_rescaled_depth,//MorphologyEx函数利用基本的膨胀和腐蚀技术，来执行更加高级形态学变换 ,此处只是膨胀变换
                       cv::MORPH_DILATE, element);
    } else {
      *dilated_rescaled_depth = *rescaled_depth;
    }
    //---step3.mask用来干啥？
    *bw_image = cv::Mat::zeros(cv_rgb_image->image.size(), CV_8UC1);

    cvtColor(cv_rgb_image->image, *bw_image, cv::COLOR_RGB2GRAY);//rgb图转灰度图

    *mask = cv::Mat::zeros(bw_image->size(), CV_8UC1);
    mask->setTo(cv::Scalar(depth_segmentation::CameraTracker::kImageRange));//mask的元素都设置为255
  }

//通过计算depth map、normal map、深度不连续图、最大距离图和最小凸度图，最终计算得到边缘图。但论文中没有提到最大距离图？
  void computeEdgeMap(const sensor_msgs::Image::ConstPtr& depth_msg,
                      const sensor_msgs::Image::ConstPtr& rgb_msg,
                      cv::Mat& rescaled_depth,
                      cv_bridge::CvImagePtr cv_rgb_image,
                      cv_bridge::CvImagePtr cv_depth_image, cv::Mat& bw_image,
                      cv::Mat& mask, cv::Mat* depth_map, cv::Mat* normal_map,
                      cv::Mat* edge_map) {
#ifdef WRITE_IMAGES
    cv::imwrite(
        std::to_string(cv_rgb_image->header.stamp.toSec()) + "_rgb_image.png",
        cv_rgb_image->image);
    cv::imwrite(
        std::to_string(cv_rgb_image->header.stamp.toSec()) + "_bw_image.png",
        bw_image);
    cv::imwrite(
        std::to_string(depth_msg->header.stamp.toSec()) + "_depth_image.png",
        rescaled_depth);
    cv::imwrite(
        std::to_string(depth_msg->header.stamp.toSec()) + "_depth_mask.png",
        mask);
#endif  // WRITE_IMAGES

#ifdef DISPLAY_DEPTH_IMAGES
    camera_tracker_.visualize(camera_tracker_.getDepthImage(), rescaled_depth);
#endif  // DISPLAY_DEPTH_IMAGES

    // Compute transform from tracker.
    if (depth_segmentation::kUseTracker) {
      if (camera_tracker_.computeTransform(bw_image, rescaled_depth, mask)) {
        publish_tf(camera_tracker_.getWorldTransform(),
                   depth_msg->header.stamp);
      } else {
        LOG(ERROR) << "Failed to compute Transform.";
      }
    }

    *depth_map = cv::Mat::zeros(depth_camera_.getWidth(),
                                depth_camera_.getHeight(), CV_32FC3);
    depth_segmenter_.computeDepthMap(rescaled_depth, depth_map);//把depth_image转化为一系列有组织的3d点集depth_map

    // Compute normal map.
    *normal_map = cv::Mat::zeros(depth_map->size(), CV_32FC3);

    if (params_.normals.method ==
            depth_segmentation::SurfaceNormalEstimationMethod::kFals ||
        params_.normals.method ==
            depth_segmentation::SurfaceNormalEstimationMethod::kSri ||
        params_.normals.method ==
            depth_segmentation::SurfaceNormalEstimationMethod::
                kDepthWindowFilter) {
      depth_segmenter_.computeNormalMap(*depth_map, normal_map);
    } else if (params_.normals.method ==
               depth_segmentation::SurfaceNormalEstimationMethod::kLinemod) {
      depth_segmenter_.computeNormalMap(cv_depth_image->image, normal_map);
    }

    // Compute depth discontinuity map.
    cv::Mat discontinuity_map = cv::Mat::zeros(
        depth_camera_.getWidth(), depth_camera_.getHeight(), CV_32FC1);
    if (params_.depth_discontinuity.use_discontinuity) {
      depth_segmenter_.computeDepthDiscontinuityMap(rescaled_depth,
                                                    &discontinuity_map);
    }

    // Compute maximum distance map.
    cv::Mat distance_map = cv::Mat::zeros(depth_camera_.getWidth(),
                                          depth_camera_.getHeight(), CV_32FC1);
    if (params_.max_distance.use_max_distance) {
      depth_segmenter_.computeMaxDistanceMap(*depth_map, &distance_map);
    }

    // Compute minimum convexity map.
    cv::Mat convexity_map = cv::Mat::zeros(depth_camera_.getWidth(),
                                           depth_camera_.getHeight(), CV_32FC1);
    if (params_.min_convexity.use_min_convexity) {
      depth_segmenter_.computeMinConvexityMap(*depth_map, *normal_map,
                                              &convexity_map);
    }

    // Compute final edge map.
    *edge_map = cv::Mat::zeros(depth_camera_.getWidth(),
                               depth_camera_.getHeight(), CV_32FC1);
    depth_segmenter_.computeFinalEdgeMap(convexity_map, distance_map,
                                         discontinuity_map, edge_map);
  }

  void imageCallback(const sensor_msgs::Image::ConstPtr& depth_msg,
                     const sensor_msgs::Image::ConstPtr& rgb_msg) {//未开启语义分割模式
    if (camera_info_ready_) {
      cv_bridge::CvImagePtr cv_rgb_image(new cv_bridge::CvImage);
      cv_rgb_image = cv_bridge::toCvCopy(rgb_msg, rgb_msg->encoding);
      if (rgb_msg->encoding == sensor_msgs::image_encodings::BGR8) {
        cv::cvtColor(cv_rgb_image->image, cv_rgb_image->image, CV_BGR2RGB);
      }

      cv_bridge::CvImagePtr cv_depth_image(new cv_bridge::CvImage);
      cv::Mat rescaled_depth, dilated_rescaled_depth, bw_image, mask, depth_map,
          normal_map, edge_map;
      preprocess(depth_msg, rgb_msg, &rescaled_depth, &dilated_rescaled_depth,
                 cv_rgb_image, cv_depth_image, &bw_image, &mask);
      if (!camera_tracker_.getRgbImage().empty() &&
              !camera_tracker_.getDepthImage().empty() ||
          !depth_segmentation::kUseTracker) {
        computeEdgeMap(depth_msg, rgb_msg, dilated_rescaled_depth, cv_rgb_image,
                       cv_depth_image, bw_image, mask, &depth_map, &normal_map,
                       &edge_map);

        cv::Mat label_map(edge_map.size(), CV_32FC1);
        cv::Mat remove_no_values =
            cv::Mat::zeros(edge_map.size(), edge_map.type());
        edge_map.copyTo(remove_no_values,
                        dilated_rescaled_depth == dilated_rescaled_depth);
        edge_map = remove_no_values;
        std::vector<depth_segmentation::Segment> segments;
        std::vector<cv::Mat> segment_masks;

        depth_segmenter_.labelMap(cv_rgb_image->image, rescaled_depth,
                                  depth_map, edge_map, normal_map, &label_map,
                                  &segment_masks, &segments);

        if (segments.size() > 0u) {
          publish_segments(segments, depth_msg->header);
        }
      }
      // Update the member images to the new images.
      // TODO(ff): Consider only doing this, when we are far enough away
      // from a frame. (Which basically means we would set a keyframe.)
      depth_camera_.setImage(rescaled_depth);
      depth_camera_.setMask(mask);
      rgb_camera_.setImage(bw_image);
    }
  }

#ifdef MASKRCNNROS_AVAILABLE
  void imageSegmentationCallback(   //TODO 获得segments最主要的函数，计时
      const sensor_msgs::Image::ConstPtr& depth_msg,
      const sensor_msgs::Image::ConstPtr& rgb_msg,
      const mask_rcnn_ros::Result::ConstPtr& segmentation_msg) {
    ros::Time begin = ros::Time::now();
    depth_segmentation::SemanticInstanceSegmentation instance_segmentation;
    semanticInstanceSegmentationFromRosMsg(segmentation_msg,   //把result msg转换为instance_segmentation里的masks和labels
                                           &instance_segmentation);

    if (camera_info_ready_) {//在收到camerInfo且调用了cameraInfoCallback()设置相机内参、里程计、segmenter之后，camera_info_ready_=true
      cv_bridge::CvImagePtr cv_rgb_image(new cv_bridge::CvImage);
      cv_rgb_image = cv_bridge::toCvCopy(rgb_msg, rgb_msg->encoding);//拷贝了rgb_msg的内容，转换为一个opencv图像
      if (rgb_msg->encoding == sensor_msgs::image_encodings::BGR8) {//转为RGB编码
        cv::cvtColor(cv_rgb_image->image, cv_rgb_image->image, CV_BGR2RGB);
      }

      cv_bridge::CvImagePtr cv_depth_image(new cv_bridge::CvImage);
      cv::Mat rescaled_depth, dilated_rescaled_depth, bw_image, mask, depth_map,
          normal_map, edge_map;
      preprocess(depth_msg, rgb_msg, &rescaled_depth, &dilated_rescaled_depth,
                 cv_rgb_image, cv_depth_image, &bw_image, &mask);
      if (!camera_tracker_.getRgbImage().empty() &&
              !camera_tracker_.getDepthImage().empty() ||
          !depth_segmentation::kUseTracker) {//rgb图像和深度图像都非空 或 !kUseTracker？
        computeEdgeMap(depth_msg, rgb_msg, dilated_rescaled_depth, cv_rgb_image,
                       cv_depth_image, bw_image, mask, &depth_map, &normal_map,
                       &edge_map);

        cv::Mat label_map(edge_map.size(), CV_32FC1);
        cv::Mat remove_no_values =
            cv::Mat::zeros(edge_map.size(), edge_map.type());
        edge_map.copyTo(remove_no_values,
                        dilated_rescaled_depth == dilated_rescaled_depth);//第二个参数是一个元素全为1的Mat，相当于直接拷贝edge_map到remove_no_values
        edge_map = remove_no_values;
        std::vector<depth_segmentation::Segment> segments;
        std::vector<cv::Mat> segment_masks;

        depth_segmenter_.labelMap(cv_rgb_image->image, rescaled_depth,//打上label并绘制轮廓
                                  instance_segmentation, depth_map, edge_map,
                                  normal_map, &label_map, &segment_masks,
                                  &segments);

        if (segments.size() > 0u) {
          publish_segments(segments, depth_msg->header);
        }
      }

      // Update the member images to the new images.
      // TODO(ff): Consider only doing this, when we are far enough away
      // from a frame. (Which basically means we would set a keyframe.)
      depth_camera_.setImage(rescaled_depth);
      depth_camera_.setMask(mask);
      rgb_camera_.setImage(bw_image);
    }
    ros::Time end = ros::Time::now();
    ros::Duration dur = end - begin;
    ROS_INFO("Publishing Segment for %lf secs", dur.toSec());
  }
#endif

  void cameraInfoCallback(//内参设定，初始化里程计和segmenter？
      const sensor_msgs::CameraInfo::ConstPtr& depth_camera_info_msg,
      const sensor_msgs::CameraInfo::ConstPtr& rgb_camera_info_msg) {
    if (camera_info_ready_) {
      return;
    }

    sensor_msgs::CameraInfo depth_info;
    depth_info = *depth_camera_info_msg;
    Eigen::Vector2d depth_image_size(depth_info.width, depth_info.height);//width和height指相机以像素为单位的分辨率

    cv::Mat K_depth = cv::Mat::eye(3, 3, CV_32FC1);//depth_info是float数组类型，转cv::Mat
    K_depth.at<float>(0, 0) = depth_info.K[0];
    K_depth.at<float>(0, 2) = depth_info.K[2];
    K_depth.at<float>(1, 1) = depth_info.K[4];
    K_depth.at<float>(1, 2) = depth_info.K[5];
    K_depth.at<float>(2, 2) = depth_info.K[8];

    depth_camera_.initialize(depth_image_size.x(), depth_image_size.y(),
                             CV_32FC1, K_depth);//深度图像是32位，rgb图是8位，是opencv 里程计要求的

    sensor_msgs::CameraInfo rgb_info;
    rgb_info = *rgb_camera_info_msg;
    Eigen::Vector2d rgb_image_size(rgb_info.width, rgb_info.height);

    cv::Mat K_rgb = cv::Mat::eye(3, 3, CV_32FC1);
    K_rgb.at<float>(0, 0) = rgb_info.K[0];
    K_rgb.at<float>(0, 2) = rgb_info.K[2];
    K_rgb.at<float>(1, 1) = rgb_info.K[4];
    K_rgb.at<float>(1, 2) = rgb_info.K[5];
    K_rgb.at<float>(2, 2) = rgb_info.K[8];

    rgb_camera_.initialize(rgb_image_size.x(), rgb_image_size.y(), CV_8UC1,
                           K_rgb);

    depth_segmenter_.initialize();//没看懂里面的max_distance、min_convexity、window_size的定义？后续看论文
    camera_tracker_.initialize(//初始化相机内参，使用opencv的里程计，这里结合了光度信息rgb和深度信息
        camera_tracker_.kCameraTrackerNames
            [camera_tracker_.CameraTrackerType::kRgbdICPOdometry]);

    camera_info_ready_ = true;
  }
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_stderrthreshold = 0;

  LOG(INFO) << "Starting depth segmentation ... ";
  ros::init(argc, argv, "depth_segmentation_node");
  DepthSegmentationNode depth_segmentation_node;
  //动态配置参数,使用dynamic_reconfig功能包配置DepthSegmenter.cfg文件，会自动生成DepthSegmenterConfig.h
  dynamic_reconfigure::Server<depth_segmentation::DepthSegmenterConfig>//创建了一个参数动态配置的服务端实例，参数配置的类型与配置文件中描述的类型相同
      reconfigure_server;                                              //该服务端实例会监听客户端的参数配置请求
  dynamic_reconfigure::Server<depth_segmentation::DepthSegmenterConfig>:://还需要定义回调函数，并将回调函数和服务端绑定。
      CallbackType dynamic_reconfigure_function;                         //当客户端请求修改参数时，服务端即可跳转到回调函数进行处理。

  dynamic_reconfigure_function = boost::bind(
      &depth_segmentation::DepthSegmenter::dynamicReconfigureCallback,
      &depth_segmentation_node.depth_segmenter_, _1, _2);//ROS回调函数有多个参数时，使用bind函数
  reconfigure_server.setCallback(dynamic_reconfigure_function);

  while (ros::ok()) {
    ros::spin();
  }

  return EXIT_SUCCESS;
}
