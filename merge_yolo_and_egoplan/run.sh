# 脚本名称: ego_planner_1uav
# 脚本描述: 单个无人机的ego_planner算法测试

gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 5; roslaunch /home/abclab/Prometheus/merge_yolo_and_egoplan/map_generator.launch swarm_num:=1; exec bash"' \
--tab -e 'bash -c "sleep 6; roslaunch /home/abclab/Prometheus/merge_yolo_and_egoplan/sitl_px4_indoor.launch uav_init_x:=-10; exec bash"' \
--tab -e 'bash -c "sleep 7; roslaunch /home/abclab/Prometheus/merge_yolo_and_egoplan/uav_control_main_indoor.launch; exec bash"' \
--tab -e 'bash -c "sleep 8; roslaunch /home/abclab/Prometheus/merge_yolo_and_egoplan/sitl_ego_planner_basic.launch; exec bash"' \
# --tab -e 'bash -c "sleep 8; roslaunch /home/abclab/Prometheus/merge_yolo_and_egoplan/a.launch; exec bash"' \
