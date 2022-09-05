#include <ros/ros.h>
#include <Eigen/Eigen>
#include <iostream>
#include <mission_utils.h>
#include <nav_msgs/Odometry.h>
#include <prometheus_msgs/DetectionInfo.h>
#include <geometry_msgs/PoseStamped.h>

#include <prometheus_msgs/UAVCommand.h>
#include <prometheus_msgs/UAVState.h>
#include <prometheus_msgs/UAVControlState.h>

#include "printf_utils.h"

using namespace std;
using namespace Eigen;

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>全 局 变 量<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
prometheus_msgs::UAVState g_UAVState;
Eigen::Vector3f g_drone_pos;
//---------------------------------------Vision---------------------------------------------
prometheus_msgs::UAVControlState g_uavcontrol_state; //目标位置[机体系下：前方x为正，右方y为正，下方z为正]
Eigen::Vector3f pos_body_frame;
Eigen::Vector3f pos_body_enu_frame;    //原点位于质心，x轴指向前方，y轴指向左，z轴指向上的坐标系
float kpx_track, kpy_track, kpz_track; //控制参数 - 比例参数
bool is_detected = false;              // 是否检测到目标标志
int num_count_vision_lost = 0;         //视觉丢失计数器
int num_count_vision_regain = 0;       //视觉丢失计数器
int Thres_vision = 0; //视觉丢失计数器阈值
Eigen::Vector3f camera_offset;
//---------------------------------------Track---------------------------------------------
float distance_to_setpoint;
Eigen::Vector3f tracking_delta;
//---------------------------------------tranport------------------------------------------
//订阅来自yolo的坐标信息
ros::Subscriber msg_from_yolo_sub;
ros::Subscriber uav_odom_sub;
//发布目标点信息到ego_planner
ros::Publisher goal_to_ego_planner_pub;
//此publisher是发布给traj_server_for_prometheus.cpp中的pub_prometheus_command订阅
ros::Publisher object_track_yaw_pub;

int uav_id;
int seq = 0;
prometheus_msgs::DetectionInfo last_msg_from_yolo;
geometry_msgs::PoseStamped last_msg_pub;
nav_msgs::Odometry uav_odom_msg;

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>回 调 函 数<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
void drone_state_cb(const prometheus_msgs::UAVState::ConstPtr &msg)
{
    g_UAVState = *msg;

    g_drone_pos[0] = g_UAVState.position[0];
    g_drone_pos[1] = g_UAVState.position[1];
    g_drone_pos[2] = g_UAVState.position[2];
}

void sub_cb(const prometheus_msgs::DetectionInfo::ConstPtr& msg_p)
{
    //--------------------判断无人机状态，暂时舍弃关于目标丢失判断的部分-----------------------
    // if (g_uavcontrol_state.control_state != prometheus_msgs::UAVControlState::COMMAND_CONTROL)
    // {
    //     PCOUT(-1, TAIL, "Waiting for enter COMMAND_CONTROL state");
    //     return;
    // }

    last_msg_from_yolo = *msg_p;
    geometry_msgs::PoseStamped target_pose;
    // 目标相对相机的位置+相机偏移，得到目标相对无人机的位置
    pos_body_frame[0] = last_msg_from_yolo.position[2] + camera_offset[0] - tracking_delta[0];
    pos_body_frame[1] = -last_msg_from_yolo.position[0] + camera_offset[1] - tracking_delta[1];
    pos_body_frame[2] = -last_msg_from_yolo.position[1] + camera_offset[2] - tracking_delta[2];

    Eigen::Matrix3f R_Body_to_ENU;

    // 获取无人机位姿，计算旋转矩阵
    R_Body_to_ENU = get_rotation_matrix(g_UAVState.attitude[0], g_UAVState.attitude[1], g_UAVState.attitude[2]);

    // 两个矩阵相乘，得到ENU系下的坐标
    pos_body_enu_frame = R_Body_to_ENU * pos_body_frame;

    //-----------------转成世界系下坐标并发送---------------------
    target_pose.header.seq = seq;
    target_pose.header.stamp =ros::Time::now();
    target_pose.header.frame_id = "world";
    target_pose.pose.position.x = pos_body_enu_frame[0] + uav_odom_msg.pose.pose.position.x;
    target_pose.pose.position.y = pos_body_enu_frame[1] + uav_odom_msg.pose.pose.position.y;
    target_pose.pose.position.z = pos_body_enu_frame[2] + uav_odom_msg.pose.pose.position.z;

    goal_to_ego_planner_pub.publish(target_pose);

    ROS_INFO("pub a goal====>(%f,%f,%f)\n",target_pose.pose.position.x,
                                            target_pose.pose.position.y,
                                            target_pose.pose.position.z);
    seq++;
    // ros::Duration(0.5).sleep();
    // 此处控制频率20Hz
}

void get_odom_cb(const nav_msgs::Odometry::ConstPtr& msg_p)
{
    uav_odom_msg = *msg_p;
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>主函数<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
int main(int argc, char **argv)
{
    ros::init(argc, argv, "my_bridge_node");
    ros::NodeHandle nh("~");
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>参数读取<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    //视觉丢失次数阈值
    nh.param<int>("Thres_vision", Thres_vision, 10);

    //相机偏移
    nh.param<float>("camera_offset_x", camera_offset[0], 0.0);
    nh.param<float>("camera_offset_y", camera_offset[1], 0.0);
    nh.param<float>("camera_offset_z", camera_offset[2], 0.0);

    nh.param<float>("tracking_delta_x", tracking_delta[0], 0.0);
    nh.param<float>("tracking_delta_y", tracking_delta[1], 0.0);
    nh.param<float>("tracking_delta_z", tracking_delta[2], 0.0);


    
    nh.param("uav_id", uav_id, 1);
    //[订阅]目标追踪信息
    msg_from_yolo_sub = nh.subscribe<prometheus_msgs::DetectionInfo>("/uav" + std::to_string(uav_id) + "/prometheus/object_detection/siamrpn_tracker",
                                                        1,
                                                        sub_cb);
    //[订阅]里程计信息
    uav_odom_sub = nh.subscribe<nav_msgs::Odometry>("/uav" + std::to_string(uav_id) + "/prometheus/odom", 
                                                    1,
                                                    get_odom_cb);
    //[订阅]无人机状态
    ros::Subscriber drone_state_sub = nh.subscribe<prometheus_msgs::UAVState>("/uav" + std::to_string(uav_id) + "/prometheus/state", 10, drone_state_cb);
    //[发布]无人机控制命令
    goal_to_ego_planner_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav" + std::to_string(uav_id) + "/prometheus/motion_planning/goal", 1);
    ROS_INFO("Node [my_bridge_node] init ok!");
    ros::spin();
    return 0;
}