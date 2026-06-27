cmake_minimum_required(VERSION 3.10)
project(livox_reflective_marker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  std_msgs
  tf2_ros
  message_generation
  livox_ros_driver2
)
find_package(Eigen3 REQUIRED)

# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
add_message_files(
  FILES
  ResourceUsage.msg
  BpuScoreEntry.msg
  BpuScores.msg
  ClusterCloud.msg
  RecognitionCommand.msg
  TargetCommand.msg
  EkfStatus.msg
  ReflectiveCandidate.msg
  ReflectiveCandidates.msg
  ReflectiveMapSnapshot.msg
  ReflectiveObservationFrame.msg
  RecognitionScoreEntry.msg
  RecognitionScores.msg
  ReflectiveRecognitionRequest.msg
  ReflectiveTargetState.msg
  ReflectiveBoardPose.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS roscpp rospy sensor_msgs geometry_msgs nav_msgs visualization_msgs std_msgs tf2_ros
                 message_runtime livox_ros_driver2
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  include
)

# ---- BPU (libdnn) -------------------------------------------------------
# libdnn is provided by the RDK X5 system; if not found the node is skipped.
find_library(DNN_LIB dnn PATHS /usr/lib /usr/local/lib)
if(DNN_LIB)
  message(STATUS "Found libdnn: ${DNN_LIB}")
  set(DNN_FOUND TRUE)
else()
  message(WARNING "libdnn not found – siamese_bpu_infer_node will NOT be built")
  set(DNN_FOUND FALSE)
endif()

# ---------------------------------------------------------------------------
# reflective_intensity_filter_node
# ---------------------------------------------------------------------------
add_executable(reflective_intensity_filter_node
  src/reflective_intensity_filter_node.cpp
)
add_dependencies(reflective_intensity_filter_node
  ${catkin_EXPORTED_TARGETS}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
)
target_link_libraries(reflective_intensity_filter_node
  ${catkin_LIBRARIES}
)

# ---------------------------------------------------------------------------
# dynamic_reflective_mapping_node
# ---------------------------------------------------------------------------
add_executable(dynamic_reflective_mapping_node
  src/dynamic_reflective_mapping_node.cpp
)
add_dependencies(dynamic_reflective_mapping_node
  ${catkin_EXPORTED_TARGETS}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
)
target_link_libraries(dynamic_reflective_mapping_node
  ${catkin_LIBRARIES}
)

# ---------------------------------------------------------------------------
# reflective_candidate_extractor_node
# ---------------------------------------------------------------------------
add_executable(reflective_candidate_extractor_node
  src/reflective_candidate_extractor_node.cpp
)
add_dependencies(reflective_candidate_extractor_node
  ${catkin_EXPORTED_TARGETS}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
)
target_link_libraries(reflective_candidate_extractor_node
  ${catkin_LIBRARIES}
)

# ---------------------------------------------------------------------------
# reflective_map_decision_node
# ---------------------------------------------------------------------------
add_executable(reflective_map_decision_node
  src/reflective_map_decision_node.cpp
)
add_dependencies(reflective_map_decision_node
  ${catkin_EXPORTED_TARGETS}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
)
target_link_libraries(reflective_map_decision_node
  ${catkin_LIBRARIES}
)

# ---------------------------------------------------------------------------
# reflective_board_pose_estimator_node
# ---------------------------------------------------------------------------
add_executable(reflective_board_pose_estimator_node
  src/reflective_board_pose_estimator_node.cpp
)
add_dependencies(reflective_board_pose_estimator_node
  ${catkin_EXPORTED_TARGETS}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
)
target_link_libraries(reflective_board_pose_estimator_node
  ${catkin_LIBRARIES}
)

# ---------------------------------------------------------------------------
# reflective_board_template_recorder_node
# ---------------------------------------------------------------------------
add_executable(reflective_board_template_recorder_node
  src/reflective_board_template_recorder_node.cpp
)
add_dependencies(reflective_board_template_recorder_node
  ${catkin_EXPORTED_TARGETS}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
)

target_link_libraries(reflective_board_template_recorder_node
  ${catkin_LIBRARIES}
)

# ---------------------------------------------------------------------------
# target_manager_node
# ---------------------------------------------------------------------------
add_executable(target_manager_node
  src/target_manager_node.cpp
)
add_dependencies(target_manager_node
  ${catkin_EXPORTED_TARGETS}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
)
target_link_libraries(target_manager_node
  ${catkin_LIBRARIES}
)

# ---------------------------------------------------------------------------
# resource_monitor_node
# ---------------------------------------------------------------------------
add_executable(resource_monitor_node src/resource_monitor_node.cpp)
add_dependencies(resource_monitor_node
  ${catkin_EXPORTED_TARGETS}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
)

target_link_libraries(resource_monitor_node
  ${catkin_LIBRARIES}
)

# ---------------------------------------------------------------------------
# siamese_bpu_infer_node  (BPU-accelerated V3 Siamese PointNet inference)
# ---------------------------------------------------------------------------
if(DNN_FOUND)
  add_executable(reflective_observation_bpu_scorer_node
    src/reflective_observation_bpu_scorer_node.cpp
  )
  add_dependencies(reflective_observation_bpu_scorer_node
    ${catkin_EXPORTED_TARGETS}
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
  )
  target_link_libraries(reflective_observation_bpu_scorer_node
    ${catkin_LIBRARIES}
    ${DNN_LIB}
  )

  add_executable(reflective_bpu_scorer_node
    src/reflective_bpu_scorer_node.cpp
  )
  add_dependencies(reflective_bpu_scorer_node
    ${catkin_EXPORTED_TARGETS}
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
  )
  target_link_libraries(reflective_bpu_scorer_node
    ${catkin_LIBRARIES}
    ${DNN_LIB}
  )

  add_executable(siamese_bpu_infer_node
    src/siamese_bpu_infer_node.cpp
  )
  add_dependencies(siamese_bpu_infer_node
    ${catkin_EXPORTED_TARGETS}
    ${${PROJECT_NAME}_EXPORTED_TARGETS}
  )
  target_link_libraries(siamese_bpu_infer_node
    ${catkin_LIBRARIES}
    ${DNN_LIB}
  )
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS
  reflective_intensity_filter_node
  dynamic_reflective_mapping_node
  reflective_candidate_extractor_node
  reflective_map_decision_node
  reflective_board_pose_estimator_node
  reflective_board_template_recorder_node
  resource_monitor_node
  target_manager_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(DNN_FOUND)
  install(TARGETS
    reflective_observation_bpu_scorer_node
    reflective_bpu_scorer_node
    siamese_bpu_infer_node
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
endif()

catkin_install_python(PROGRAMS
  scripts/analyze_target_pose_jitter.py
  scripts/auto_label_board_candidates.py
  scripts/clicked_pointcloud2_reflectivity.py
  scripts/pointcloud2_intensity_filter.py
  scripts/relabel_auto_board_candidate.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config msg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
