cmake_minimum_required(VERSION 3.16)
project(openeta_rm75_robotiq2f85_sim)
find_package(ament_cmake REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-sim8 REQUIRED)
find_package(gz-transport13 REQUIRED)

add_library(openeta_m3_adhesion_system SHARED src/m3_adhesion_system.cpp)
target_compile_features(openeta_m3_adhesion_system PRIVATE cxx_std_17)
target_link_libraries(openeta_m3_adhesion_system PRIVATE
  gz-plugin2::gz-plugin2
  gz-sim8::gz-sim8
  gz-transport13::gz-transport13
)

set(VENDOR_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../assets/robotiq_2f85_vendor")
if(NOT EXISTS "${VENDOR_ROOT}/asset_manifest.json")
  message(FATAL_ERROR "MODEL_ASSET_NOT_FOUND: vendor closure missing")
endif()
install(DIRECTORY launch urdf config worlds DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/gripper_action_adapter.py DESTINATION lib/${PROJECT_NAME})
install(TARGETS openeta_m3_adhesion_system DESTINATION lib)
install(DIRECTORY "${VENDOR_ROOT}/meshes/" DESTINATION share/${PROJECT_NAME}/meshes)
install(DIRECTORY "${VENDOR_ROOT}/urdf/" DESTINATION share/${PROJECT_NAME}/vendor/robotiq)
install(FILES "${VENDOR_ROOT}/LICENSE" "${VENDOR_ROOT}/NOTICE" DESTINATION share/${PROJECT_NAME}/vendor)
ament_package()
