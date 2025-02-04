#!/bin/bash

# 脚本名称: ego_planner_1uav
# 脚本描述: 单个无人机的ego_planner算法测试

# 激活conda环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate prometheus_python3

gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 1; roslaunch ./map_generator.launch swarm_num:=1; exec bash"' \
--tab -e 'bash -c "sleep 1; roslaunch ./sitl_px4_indoor.launch uav_init_x:=-10; exec bash"' \
--tab -e 'bash -c "sleep 1; roslaunch ./uav_control_main_indoor.launch; exec bash"' \
--tab -e 'bash -c "sleep 2; roslaunch ./sitl_ego_planner_basic.launch; exec bash"' \
--tab -e 'bash -c "sleep 2; source devel/setup.bash; roslaunch position_transpoter position_transpoter.launch; exec bash"' \
--tab -e 'bash -c "sleep 2; cd /home/abclab/Prometheus/Modules/object_detection_yolov5tensorrt; python yolov5_trt_ros.py  --image_topic /prometheus/sensor/monocular_front/image_raw;exec bash"' \
--tab -e 'bash -c "sleep 3; cd /home/abclab/Prometheus/Modules/object_detection/py_nodes/yolov5_tensorrt_client; python yolov5_tensorrt_client.py; exec bash"' \
--tab -e 'bash -c "sleep 3; source devel/setup.bash; rosrun gazebo_extra_plugins keyboard_actor_pyqt.py; exec bash"'\
# --tab -e 'bash -c "sleep 3; roslaunch ./detect.launch; exec bash"' \