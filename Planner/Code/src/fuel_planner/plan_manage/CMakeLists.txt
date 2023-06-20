cmake_minimum_required(VERSION 2.8.3)
project(plan_manage)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_CXX_STANDARD 14)

set(Torch_DIR /home/albert/3rdparty/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  quadrotor_msgs
  poly_traj
  plan_env
  path_searching
  lkh_tsp_solver
  bspline
  bspline_opt
  traj_utils
  active_perception
  cv_bridge
  tf
)


catkin_package(
 INCLUDE_DIRS include
 LIBRARIES plan_manage
 CATKIN_DEPENDS plan_env path_searching bspline bspline_opt active_perception traj_utils)

include_directories(
  include
  SYSTEM
  ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${TORCH_LIBRARIES}
)


add_executable(fast_planner_node
  src/fast_planner_node.cpp 
  src/kino_replan_fsm.cpp
  src/topo_replan_fsm.cpp
  test/local_explore_fsm.cpp
  src/planner_manager.cpp
  src/planner_manager_dev.cpp
  )
target_link_libraries(fast_planner_node 
  ${catkin_LIBRARIES}
  dw
  )

add_library(plan_manage 
  src/planner_manager.cpp
  src/planner_manager_dev.cpp
  src/local_planner.cpp
  )
target_link_libraries(plan_manage ${catkin_LIBRARIES}
${TORCH_LIBRARIES})

add_executable(traj_server src/traj_server.cpp
src/trajectory_generator_waypoint.cpp)
target_link_libraries(traj_server ${catkin_LIBRARIES} -ldw)
# add_dependencies(traj_server ${${PROJECT_NAME}_EXPORTED_TARGETS})

# benchmark node
# add_executable(compare_topo
#   src/compare_topo.cpp 
#   src/planner_manager.cpp
#   )
# target_link_libraries(compare_topo 
#   ${catkin_LIBRARIES}
#   )

# add_executable(rotation test/rotation.cpp)
# target_link_libraries(rotation ${catkin_LIBRARIES})

add_executable(proc_msg 
    test/process_msg.cpp
)
target_link_libraries( proc_msg
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    )  

add_executable(proc_msg2 
    test/process_msg2.cpp
)
target_link_libraries( proc_msg2
    ${catkin_LIBRARIES} 
    ${PCL_LIBRARIES}
    )  
