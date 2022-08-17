cmake_minimum_required(VERSION 2.8.3)
project(prometheus_detection)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)

# set(OpenCV_DIR /usr/local/share/OpenCV)

# Define path of darknet folder here.
find_path(DARKNET_PATH
  NAMES "Makefile"
  HINTS "${CMAKE_CURRENT_SOURCE_DIR}/src/darknet_lib/")
message("Darknet path dir = ${DARKNET_PATH}")
add_definitions(-DDARKNET_FILE_PATH="${DARKNET_PATH}")

# Find CUDA
find_package(CUDA QUIET)
if (CUDA_FOUND)
  find_package(CUDA REQUIRED)
  message(STATUS "CUDA Version: ${CUDA_VERSION_STRINGS}")
  message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
  set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=[sm_50,compute_50]
    -gencode arch=compute_52,code=[sm_52,compute_52]
    -gencode arch=compute_61,code=sm_61
    -gencode arch=compute_62,code=sm_62
  )
  add_definitions(-DGPU)
else()
  list(APPEND LIBRARIES "m")
endif()

# Find X11
MESSAGE ( STATUS "Searching for X11..." )
FIND_PACKAGE ( X11 REQUIRED )
IF ( X11_FOUND )
  INCLUDE_DIRECTORIES ( ${X11_INCLUDE_DIR} )
  LINK_LIBRARIES ( ${X11_LIBRARIES} )
  MESSAGE ( STATUS " X11_INCLUDE_DIR: " ${X11_INCLUDE_DIR} )
  MESSAGE ( STATUS " X11_LIBRARIES: " ${X11_LIBRARIES} )
ENDIF ( X11_FOUND )


## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    message_generation
    rostime
    roscpp
    rosconsole
    roscpp_serialization
    sensor_msgs
    image_transport
    nodelet
    cv_bridge
    actionlib
    prometheus_msgs
    tf2_eigen
    tf
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS
    thread
    date_time
)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# Enable OPENCV in darknet
add_definitions(-DOPENCV)
add_definitions(-O4 -g)

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs  # Or other packages containing msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need

# catkin_package(
#    CATKIN_DEPENDS message_runtime std_msgs
# )
catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    roscpp
    actionlib
    rospy
    std_msgs
    prometheus_msgs
    image_transport
    nodelet
  DEPENDS
    Boost
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  src/kcf_tracker_lib
  # src/ellipse_lib
  src/darknet_lib/src
  src/darknet_lib/include
  src/darknet_ros/include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INUCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ../common/include
)

set(PROJECT_LIB_FILES
    src/darknet_ros/src/YoloObjectDetector.cpp                    src/darknet_ros/src/image_interface.cpp
)

set(DARKNET_CORE_FILES
    ${DARKNET_PATH}/src/activation_layer.c        ${DARKNET_PATH}/src/im2col.c
    ${DARKNET_PATH}/src/activations.c             ${DARKNET_PATH}/src/image.c
    ${DARKNET_PATH}/src/avgpool_layer.c           ${DARKNET_PATH}/src/layer.c
    ${DARKNET_PATH}/src/batchnorm_layer.c         ${DARKNET_PATH}/src/list.c
    ${DARKNET_PATH}/src/blas.c                    ${DARKNET_PATH}/src/local_layer.c
    ${DARKNET_PATH}/src/box.c                     ${DARKNET_PATH}/src/lstm_layer.c
    ${DARKNET_PATH}/src/col2im.c                  ${DARKNET_PATH}/src/matrix.c
    ${DARKNET_PATH}/src/connected_layer.c         ${DARKNET_PATH}/src/maxpool_layer.c
    ${DARKNET_PATH}/src/convolutional_layer.c     ${DARKNET_PATH}/src/network.c
    ${DARKNET_PATH}/src/cost_layer.c              ${DARKNET_PATH}/src/normalization_layer.c
    ${DARKNET_PATH}/src/crnn_layer.c              ${DARKNET_PATH}/src/option_list.c
    ${DARKNET_PATH}/src/crop_layer.c              ${DARKNET_PATH}/src/parser.c
    ${DARKNET_PATH}/src/cuda.c                    ${DARKNET_PATH}/src/region_layer.c
    ${DARKNET_PATH}/src/data.c                    ${DARKNET_PATH}/src/reorg_layer.c
    ${DARKNET_PATH}/src/deconvolutional_layer.c   ${DARKNET_PATH}/src/rnn_layer.c
    ${DARKNET_PATH}/src/demo.c                    ${DARKNET_PATH}/src/route_layer.c
    ${DARKNET_PATH}/src/detection_layer.c         ${DARKNET_PATH}/src/shortcut_layer.c
    ${DARKNET_PATH}/src/dropout_layer.c           ${DARKNET_PATH}/src/softmax_layer.c
    ${DARKNET_PATH}/src/gemm.c                    ${DARKNET_PATH}/src/tree.c
    ${DARKNET_PATH}/src/gru_layer.c               ${DARKNET_PATH}/src/utils.c
    ${DARKNET_PATH}/src/upsample_layer.c          ${DARKNET_PATH}/src/logistic_layer.c
    ${DARKNET_PATH}/src/l2norm_layer.c            ${DARKNET_PATH}/src/yolo_layer.c
    ${DARKNET_PATH}/src/iseg_layer.c              ${DARKNET_PATH}/src/image_opencv.cpp

    ${DARKNET_PATH}/examples/art.c                ${DARKNET_PATH}/examples/lsd.c
    ${DARKNET_PATH}/examples/nightmare.c          ${DARKNET_PATH}/examples/instance-segmenter.c
    ${DARKNET_PATH}/examples/captcha.c            ${DARKNET_PATH}/examples/regressor.c
    ${DARKNET_PATH}/examples/cifar.c              ${DARKNET_PATH}/examples/rnn.c
    ${DARKNET_PATH}/examples/classifier.c         ${DARKNET_PATH}/examples/segmenter.c
    ${DARKNET_PATH}/examples/coco.c               ${DARKNET_PATH}/examples/super.c
    ${DARKNET_PATH}/examples/darknet.c            ${DARKNET_PATH}/examples/tag.c
    ${DARKNET_PATH}/examples/detector.c           ${DARKNET_PATH}/examples/yolo.c
    ${DARKNET_PATH}/examples/go.c
)

set(DARKNET_CUDA_FILES
    ${DARKNET_PATH}/src/activation_kernels.cu     ${DARKNET_PATH}/src/crop_layer_kernels.cu
    ${DARKNET_PATH}/src/avgpool_layer_kernels.cu  ${DARKNET_PATH}/src/deconvolutional_kernels.cu
    ${DARKNET_PATH}/src/blas_kernels.cu           ${DARKNET_PATH}/src/dropout_layer_kernels.cu
    ${DARKNET_PATH}/src/col2im_kernels.cu         ${DARKNET_PATH}/src/im2col_kernels.cu
    ${DARKNET_PATH}/src/convolutional_kernels.cu  ${DARKNET_PATH}/src/maxpool_layer_kernels.cu
)


if (CUDA_FOUND)

  link_directories(
    /usr/local/cuda/lib64
  )

  cuda_add_library(darknet_ros_lib
    ${PROJECT_LIB_FILES} ${DARKNET_CORE_FILES}
    ${DARKNET_CUDA_FILES}
  )

  target_link_libraries(darknet_ros_lib
    cuda
    cudart
    cublas
    curand
  )

  cuda_add_executable(darknet_ros
    src/darknet_ros/src/yolo_object_detector_node.cpp
  )

  cuda_add_library(darknet_ros_nodelet
    src/darknet_ros/src/yolo_object_detector_nodelet.cpp
  )

else()

  add_library(darknet_ros_lib
  ${PROJECT_LIB_FILES} ${DARKNET_CORE_FILES}
  )

  add_executable(darknet_ros
    src/darknet_ros/src/yolo_object_detector_node.cpp
  )

  add_library(darknet_ros_nodelet
  src/yolo_object_detector_nodelet.cpp
  )

endif()

target_link_libraries(darknet_ros_lib
    m
    pthread
    stdc++
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${OpenCV_LIBS}
)

target_link_libraries(darknet_ros
  darknet_ros_lib
)

target_link_libraries(darknet_ros_nodelet
  darknet_ros_lib
)

add_dependencies(darknet_ros_lib
  darknet_ros_msgs_generate_messages_cpp
)

install(TARGETS darknet_ros_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS darknet_ros
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY include/darknet_ros/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

install(DIRECTORY config launch yolo_network_config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

## Declare a C++ library

## Libs
add_library(kcftrackerlib
    src/kcf_tracker_lib/kcftracker.hpp
    src/kcf_tracker_lib/ffttools.hpp
    src/kcf_tracker_lib/fhog.hpp
    src/kcf_tracker_lib/labdata.hpp
    src/kcf_tracker_lib/recttools.hpp
    src/kcf_tracker_lib/tracker.h
    src/kcf_tracker_lib/kcftracker.cpp
    src/kcf_tracker_lib/fhog.cpp
)
target_link_libraries(kcftrackerlib ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})

## Nodes
add_executable(web_cam
    cpp_nodes/web_cam.cpp
)
target_link_libraries(web_cam ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(kcf_tracker
    cpp_nodes/kcf_tracker.cpp
)
target_link_libraries(kcf_tracker ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} kcftrackerlib yaml-cpp)

add_executable(aruco_det
    cpp_nodes/aruco_det.cpp
)
target_link_libraries(aruco_det ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} yaml-cpp Eigen3::Eigen)

add_executable(create_board_charuco
    cpp_nodes/create_board_charuco.cpp
)
target_link_libraries(create_board_charuco ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(calibrate_camera_charuco
    cpp_nodes/calibrate_camera_charuco.cpp
)
target_link_libraries(calibrate_camera_charuco ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS})

add_executable(aruco_navigation_switch
    cpp_nodes/aruco_navigation_switch.cpp
)
target_link_libraries(aruco_navigation_switch ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

add_executable(markers_landpad_det
    cpp_nodes/markers_landpad_det.cpp
)
target_link_libraries(markers_landpad_det ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)

# add_executable(ellipse_det
#     cpp_nodes/ellipse_det.cpp
#     src/ellipse_detector/ellipse_detector.cpp
#     src/ellipse_detector/ellipse_detector.h
# )
# target_include_directories(ellipse_det PUBLIC
#     src/ellipse_detector/
# )
# target_link_libraries(ellipse_det ${catkin_LIBRARIES} ${OpenCV_LIBS} yaml-cpp)


if(CATKIN_ENABLE_TESTING)
  # Download yolov3-tiny.weights
  set(PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/darknet_ros/yolo_network_config/weights")
  set(FILE "${PATH}/yolov3-tiny.weights")
  message(STATUS "Checking and downloading yolov3-tiny.weights if needed ...")
  if (NOT EXISTS "${FILE}")
    message(STATUS "... file does not exist. Downloading now ...")
    execute_process(COMMAND wget http://jario.ren/upload/yolov3-tiny.weights -P ${PATH})
  endif()
  # Download yolov3-tiny-BDT-best.weights
  set(PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/darknet_ros/yolo_network_config/weights")
  set(FILE "${PATH}/yolov3-tiny-BDT-best.weights")
  message(STATUS "Checking and downloading yolov3-tiny-BDT-best.weights if needed ...")
  if (NOT EXISTS "${FILE}")
    message(STATUS "... file does not exist. Downloading now ...")
    execute_process(COMMAND wget http://jario.ren/upload/yolov3-tiny-BDT-best.weights -P ${PATH})
  endif()
  # Download SiamRPNVOT.model
  set(PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/siam_rpn_lib")
  set(FILE "${PATH}/SiamRPNVOT.model")
  message(STATUS "Checking and downloading SiamRPNVOT.model if needed ...")
  if (NOT EXISTS "${FILE}")
    message(STATUS "... file does not exist. Downloading now ...")
    execute_process(COMMAND wget http://jario.ren/upload/SiamRPNVOT.model -P ${PATH})
  endif()
endif()

#########################
###   CLANG TOOLING   ###
#########################
find_package(cmake_clang_tools QUIET)
if (cmake_clang_tools_FOUND)
  message(STATUS "Run clang tooling")
  add_clang_tooling(
    TARGETS ${PROJECT_NAME}
    SOURCE_DIRS ${CMAKE_CURRENT_LIST_DIR}/src ${CMAKE_CURRENT_LIST_DIR}/include ${CMAKE_CURRENT_LIST_DIR}/test
    CT_HEADER_DIRS ${CMAKE_CURRENT_LIST_DIR}/include
    CF_WERROR
  )
endif (cmake_clang_tools_FOUND)