# open_manipulator_x

ROS package for Open Manipulator Robot to provide functionality of GYM like environment for Reinforcement Learning

# Requirements

1. OpenMANIPULATOR-X ros packages
  - Link: https://emanual.robotis.com/docs/en/platform/openmanipulator_x/quick_start_guide/
  

# Basic Fuctionalities
1. Object Detector
  - uses pyrealsense2 to access intel D455 depth camera to get RGB and Depth image.
  - Detect object ([u,v] i.e. center of object in image coordinates) in the RGB image (simple pink/red color object based on HSV thresholding)
  - Backproject (u,v) to (X,Y,Z)_cam using depth
  - Transfomrs (X,Y,Z)_cam to (X,Y,Z)_wolrd using a Chekerboard pattern placed in the world acting as origin of world cordinate
  - Publishes object_position to /object_pos topoic which can be subscribed by other nodes
