// Copyright (c) 2019, ASL, ETH Zurich, Switzerland
// Licensed under the BSD 3-Clause License (see LICENSE for details)

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <ros/ros.h>

#include "global_segment_map_node/controller.h"

int main(int argc, char** argv) {
  ros::init(argc, argv, "gsm_node");
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);//解析命令行参数。第三个参数false，则解析后的argv和argc会被保留，但是注意函数会调整argv中的顺序。
  google::InstallFailureSignalHandler();// 当程序在某些信号崩溃时，它将保存出错信息

  ros::NodeHandle node_handle;
  ros::NodeHandle node_handle_private("~");

  voxblox::voxblox_gsm::Controller* controller;

  std::cout << endl
            << "Voxblox++ Copyright (C) 2016-2020 ASL, ETH Zurich." << endl
            << endl;

  controller = new voxblox::voxblox_gsm::Controller(&node_handle_private);

  ros::ServiceServer reset_map_srv;
  controller->advertiseResetMapService(&reset_map_srv);//创建重置地图的server实例

  ros::ServiceServer toggle_integration_srv;
  controller->advertiseToggleIntegrationService(&toggle_integration_srv);//intergration开关

  ros::Subscriber segment_point_cloud_sub;
  controller->subscribeSegmentPointCloudTopic(&segment_point_cloud_sub);// 订阅/depth_segmentation_node/object_segment

//创建controller时，publish_scene_map/mesh/object_bbox=false, 即下面四个话题默认不发布
  if (controller->publish_scene_map_) {
    controller->advertiseMapTopic();
  }

  if (controller->publish_scene_mesh_) {
    controller->advertiseSceneMeshTopic();
    controller->advertiseSceneCloudTopic();
  }

  if (controller->publish_object_bbox_) {
    controller->advertiseBboxTopic();
  }

  ros::ServiceServer get_map_srv;
  controller->advertiseGetMapService(&get_map_srv);

  ros::ServiceServer generate_mesh_srv;
  controller->advertiseGenerateMeshService(&generate_mesh_srv);

  ros::ServiceServer get_scene_pointcloud;
  controller->advertiseGetScenePointcloudService(&get_scene_pointcloud);

  ros::ServiceServer save_segments_as_mesh_srv;
  controller->advertiseSaveSegmentsAsMeshService(&save_segments_as_mesh_srv);

  ros::ServiceServer extract_instances_srv;
  ros::ServiceServer get_list_semantic_instances_srv;
  ros::ServiceServer get_instance_bounding_box_srv;

  if (controller->enable_semantic_instance_segmentation_) {//默认为true
    controller->advertiseExtractInstancesService(&extract_instances_srv);
    controller->advertiseGetListSemanticInstancesService(
        &get_list_semantic_instances_srv);
    controller->advertiseGetAlignedInstanceBoundingBoxService(
        &get_instance_bounding_box_srv);
  }

  // Spinner that uses a number of threads equal to the number of cores.
  ros::AsyncSpinner spinner(0);
  spinner.start();
  ros::waitForShutdown();

  LOG(INFO) << "Shutting down.";
  return 0;
}
