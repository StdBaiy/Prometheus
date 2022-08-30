#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <prometheus_msgs/DetectionInfo.h>
#include <geometry_msgs/PoseStamped.h>

//订阅来自yolo的坐标信息
ros::Subscriber msg_from_yolo_sub;
ros::Subscriber uav_odom_sub;
//发布目标点信息到ego_planner
ros::Publisher goal_to_ego_planner_pub;


ros::Timer timer;

int uav_id;
int seq = 0;
bool yolo_msg_flag = false;
bool has_pub_before = false;
prometheus_msgs::DetectionInfo last_msg_from_yolo;
geometry_msgs::PoseStamped last_msg_pub;
nav_msgs::Odometry uav_odom_msg;


void sub_cb(const prometheus_msgs::DetectionInfo::ConstPtr& msg_p)
{
    yolo_msg_flag = true;
    last_msg_from_yolo = *msg_p;
    // ROS_INFO("Get msg from yolo5===>(%f,%f,%f)",last_msg_from_yolo.position[2],
    //                                             last_msg_from_yolo.position[1],
    //                                             last_msg_from_yolo.position[0]);
}

void get_odom_cb(const nav_msgs::Odometry::ConstPtr& msg_p)
{
    uav_odom_msg = *msg_p;
}

void time_cb(const ros::TimerEvent &event)
{
    ROS_INFO("time_cb");
    if(!yolo_msg_flag)return;

    geometry_msgs::PoseStamped target_pose;
    target_pose.header.seq = seq;
    target_pose.header.stamp =ros::Time::now();
    target_pose.header.frame_id = "world";
    target_pose.pose.position.x = last_msg_from_yolo.position[2] + uav_odom_msg.pose.pose.position.x;
    target_pose.pose.position.y = -last_msg_from_yolo.position[0] + uav_odom_msg.pose.pose.position.y;
    target_pose.pose.position.z = -last_msg_from_yolo.position[1] + uav_odom_msg.pose.pose.position.z;
    target_pose.pose.orientation.x = last_msg_from_yolo.attitude_q[0];
    target_pose.pose.orientation.y = last_msg_from_yolo.attitude_q[1];
    target_pose.pose.orientation.z = last_msg_from_yolo.attitude_q[2];
    target_pose.pose.orientation.w = last_msg_from_yolo.attitude_q[3];
    goal_to_ego_planner_pub.publish(target_pose);

    //是否之前发送过目标点，如果未发送过则直接发送
    //未发送过则判断和上次目标点距离是否大于某个阈值，距离远才发送
    if(!has_pub_before)
    {
        goal_to_ego_planner_pub.publish(target_pose);
        ROS_INFO("pub a goal====>(%f,%f,%f)\n",target_pose.pose.position.x,
                                                target_pose.pose.position.y,
                                                target_pose.pose.position.z);
        seq++;
    }
    else
    {
        float distance,x_d,y_d,z_d;
        x_d = last_msg_from_yolo.position[0] - last_msg_pub.pose.position.x;
        y_d = last_msg_from_yolo.position[1] - last_msg_pub.pose.position.y;
        z_d = last_msg_from_yolo.position[2] - last_msg_pub.pose.position.z;
        distance = (x_d*x_d)+(y_d*y_d)+(z_d*z_d);

        if(distance>(0.1*0.1))
        {
            goal_to_ego_planner_pub.publish(target_pose);
            ROS_INFO("pub a goal====>(%f,%f,%f)\n",target_pose.pose.position.x,
                                                    target_pose.pose.position.y,
                                                    target_pose.pose.position.z);
            seq++;
        }
    }
        

}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "my_bridge_node");
    ros::NodeHandle nh("~");

    
    nh.param("uav_id", uav_id, 1);
    msg_from_yolo_sub = nh.subscribe<prometheus_msgs::DetectionInfo>("/uav" + std::to_string(uav_id) + "/prometheus/object_detection/siamrpn_tracker",
                                                        1,
                                                        sub_cb);
    uav_odom_sub = nh.subscribe<nav_msgs::Odometry>("/uav" + std::to_string(uav_id) + "/prometheus/odom", 
                                                    1,
                                                    get_odom_cb);

    goal_to_ego_planner_pub = nh.advertise<geometry_msgs::PoseStamped>("/uav" + std::to_string(uav_id) + "/prometheus/motion_planning/goal", 1);
    timer = nh.createTimer(ros::Duration(1.0), time_cb);
    ROS_INFO("Node [my_bridge_node] init ok!");
    ros::spin();
    return 0;
}