<img src="https://z3.ax1x.com/2021/05/05/gKVnHJ.png" alt="prometheus logo" align="right" height="70" />

# Prometheus - 自主无人机开源项目

[[English Readme]](https://github.com/amov-lab/Prometheus/blob/master/README_EN.md)

**Prometheus**是希腊神话中最具智慧的神明之一，希望本项目能为无人机研发带来无限的智慧与光明。

## 修改说明
- 本项目已经经过编译，如果需要移植，需要先用原始的Prometheus项目进行编译，然后把/Prometheus文件夹换成本文件夹
- 本机环境：
  - Ubuntu20.04
  - OpenCV4.3.0（在object_detection的CMakeLists.txt中手动指定了OpenCV的路径）
  - NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7
  - 目标检测模块需要conda对应的python环境，具体见[Prometheus使用手册](https://wiki.amovlab.com/public/prometheus-wiki/)
- YoloV5部分是个内嵌的git仓库，为了方便我把它变成了普通文件夹
- 我把原始项目的darknet_ros部分换成了github上原作者的项目，同时修改了CMakeLists.txt
- 部分需要用到旧版本OpenCV的模块被我注释了，比如椭圆检测部分，实际不影响其他demo运行
***
- 修改了部分sdf和world文件内容（在我的机子上无法通过include直接引用sdf的话题，推测为gazebo的问题）
- 添加了若干用于调试的脚本
- 训练了自己的yolov5模型
***
- 结合了追踪和避障模块，整合了启动脚本于merge_yolo_and_egoplan/
- 修改了ego的状态机，使之能边规划边飞行
- 修改了traj_server_for_prometheus.cpp的pub_prometheus_command函数，使飞机的姿态角始终对准目标


## 项目总览

Prometheus是一套开源的**自主无人机软件系统平台**，为无人机的智能与自主飞行提供**全套解决方案**。本项目基于[PX4开源飞控固件](https://docs.px4.io/main/zh/index.html)和[ROS机器人操作系统](https://wiki.ros.org/)，旨在为无人机开发者配套成熟可用的**机载电脑端**软件系统，提供更加简洁快速的开发体验。目前已集成**控制**、**规划**及**目标检测**等研究方向，提供多个功能demo，并配套有Amovlab仿真组件。

 - Github：https://github.com/amov-lab/Prometheus
 - Gitee：https://gitee.com/amovlab/Prometheus
 - **开源项目，维护不易，还烦请点一个star收藏，谢谢支持！**

## 快速入门

 - 安装及使用：[Prometheus使用手册](https://wiki.amovlab.com/public/prometheus-wiki/)
    - 需掌握C语言基础（大部分程序为C语言，部分模块有少量C++和python的语法）。
    - 纯新手入门者建议先自学ROS官网教程。
    - PX4飞控代码可不掌握，但需要掌握基本概念及基本操作。

 - 答疑及交流：
    - 答疑论坛（官方定期答疑，推荐）：[阿木社区-Prometheus问答专区](https://bbs.amovlab.com/forum.php?mod=forumdisplay&fid=101)
    - 添加微信jiayue199506（备注消息：Prometheus）进入Prometheus自主无人机交流群。
    - B站搜索并关注“阿木社区”，开发团队定期直播答疑。

## 进阶学习

 - Promehteus自主无人机二次开发课程： 本课程偏重本项目中的基础知识和操作实践，适合本项目入门者。课程报名[请戳这里](https://bbs.amovlab.com/plugin.php?id=zhanmishu_video:video&mod=video&cid=43)。
 - 更多进阶学习资料正在紧张筹备中。

## 硬件产品

- **仿真必备遥控器**

  [点击购买](https://item.taobao.com/item.htm?spm=a1z10.5-c-s.w4002-22617251051.13.2ffa3e75uvfxuB&id=612837659406)

- **Prometheus二次开发无人机平台**   

  - 整机平台：[Prometheus  230](https://mp.weixin.qq.com/s/Tc1VPPGdA3-rw1glKwjIRg)、[Prometheus 450](https://mp.weixin.qq.com/s/LdtmLQ2eYUwg-pRklMXL8w)、[Prometheus  600](https://mp.weixin.qq.com/s/LgHU2E34d37wiX2jcgSPPA)
  - 丰富的文档资料，详情[请戳这里](https://wiki.amovlab.com/public/prometheuswiki/) 。
  - 售后服务与技术指导。
  - 免费赠送 成都线下工程实训课程 及 Promehteus自主无人机二次开发课程购买折扣。

- **Prometheus二次开发配套硬件**   

  - 关键配件：[Allspark微型边缘计算机](https://mp.weixin.qq.com/s/EF7wRWPuazYwUlXaUr2g_Q)、[超小型G1吊舱](https://mp.weixin.qq.com/s/ddtYEVJkY8TpH47NjUxKXQ)

  - 通讯配件：[minihomer](https://mp.weixin.qq.com/s/7sasXY_8S1DqYsrgs0U5NA)、[自组网homer](https://mp.weixin.qq.com/s/5ap0EeeWdk4IcDGNv1IaZw)、

  - 其他配件：[Nora+飞控](https://item.taobao.com/item.htm?spm=a1z10.5-c-s.w4002-22617251051.24.73186f33qJobSF&id=676347508327)

  - 其他配套硬件：机架、机载电脑、双目、激光雷达等无人机二次开发配套硬件

    请关注 [阿木实验室淘宝店](https://shop142114972.taobao.com/?spm=a230r.7195193.1997079397.2.67d03d8dJQgFRW)　或　[阿木实验室京东旗舰店](https://mall.jd.com/index-10260560.html?from=pc)。

## Prometheus校园赞助计划
 - 奖励使用Prometheus进行实验并发表相关论文的学生科研工作者。
 
   > @misc{Prometheus, author = "Amovlab", title = "Prometheus autonomous UAV opensource project", howpublished = "\url{https://github.com/amov-lab/Prometheus }", } 
 
 - 奖励为Prometheus提供新功能或帮忙进行测试的学生开发者。

- 详情[请戳这里](https://mp.weixin.qq.com/s/zU-iXMKh0An-v6vZXH_Rmg) ！

## 联系我们

- 公司官网：[阿木实验室](https://www.amovlab.com)

- 项目合作、无人机软硬件定制，请添加微信“yanyue199506”（备注消息：Prometheus定制）。

## 版权声明

 - 本项目受 Apache License 2.0 协议保护。点击 [LICENSE](https://wiki.amovlab.com/public/prometheus-wiki/Prometheus-%E8%87%AA%E4%B8%BB%E6%97%A0%E4%BA%BA%E6%9C%BA%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90License%E5%8F%8A%E7%89%88%E6%9D%83%E5%A3%B0%E6%98%8E.html)了解更多
 - 本项目仅限个人使用，请勿用于商业用途。
 - 如利用本项目进行营利活动，阿木实验室将追究侵权行为。
