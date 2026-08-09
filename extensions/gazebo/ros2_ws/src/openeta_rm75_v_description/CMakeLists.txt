cmake_minimum_required(VERSION 3.16)
project(openeta_rm75_v_description)
find_package(ament_cmake REQUIRED)

set(ASSET_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../assets/rm75_6fb_v_vendor")
if(NOT EXISTS "${ASSET_ROOT}/asset_manifest.json")
  message(FATAL_ERROR "MODEL_ASSET_NOT_FOUND: ${ASSET_ROOT}")
endif()

install(DIRECTORY "${ASSET_ROOT}/meshes/" DESTINATION share/${PROJECT_NAME}/meshes)
install(DIRECTORY "${ASSET_ROOT}/urdf/" DESTINATION share/${PROJECT_NAME}/urdf)
install(FILES "${ASSET_ROOT}/LICENSE" "${ASSET_ROOT}/NOTICE"
        DESTINATION share/${PROJECT_NAME})
ament_package()
