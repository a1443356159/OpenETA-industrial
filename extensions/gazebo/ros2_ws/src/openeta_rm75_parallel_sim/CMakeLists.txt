cmake_minimum_required(VERSION 3.16)
project(openeta_rm75_parallel_sim)
find_package(ament_cmake REQUIRED)

install(DIRECTORY launch urdf config worlds DESTINATION share/${PROJECT_NAME})
ament_package()
