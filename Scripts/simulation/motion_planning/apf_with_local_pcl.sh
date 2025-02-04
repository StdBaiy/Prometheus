# 脚本名称: apf_with_local_pcl
# 脚本描述: 单个无人机的apf算法测试(局部点云)

gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 5; roslaunch prometheus_simulator_utils map_generator.launch; exec bash"' \
--tab -e 'bash -c "sleep 6; roslaunch prometheus_gazebo sitl_px4_indoor.launch uav_init_x:=-10; exec bash"' \
--tab -e 'bash -c "sleep 7; roslaunch prometheus_uav_control uav_control_main_indoor.launch; exec bash"' \
--tab -e 'bash -c "sleep 8; roslaunch prometheus_local_planner sitl_apf_with_local_point.launch; exec bash"' \