cmake_minimum_required(VERSION 3.5)
project(vesc_driver)

# Set minimum C++ standard to C++14
if(NOT "${CMAKE_CXX_STANDARD_COMPUTED_DEFAULT}")
  message(STATUS "Changing CXX_STANDARD from C++98 to C++14")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
elseif("${CMAKE_CXX_STANDARD_COMPUTED_DEFAULT}" STREQUAL "98")
  message(STATUS "Changing CXX_STANDARD from C++98 to C++14")
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

find_package(Threads)

###########
## Build ##
###########

# node library
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/vesc_driver.cpp
  src/vesc_interface.cpp
  src/vesc_packet.cpp
  src/vesc_packet_factory.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${CMAKE_THREAD_LIBS_INIT}
)

# 명시적으로 종속 라이브러리 연결(중복이어도 안전)
ament_target_dependencies(${PROJECT_NAME}
  serial_driver
  io_context
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN vesc_driver::VescDriver
  EXECUTABLE ${PROJECT_NAME}_node
)

ament_auto_add_executable(
  vesc_device_namer
  src/vesc_device_namer.cpp
  src/vesc_device_uuid_lookup.cpp
)

# 실행 파일에도 동일 종속 명시
ament_target_dependencies(vesc_device_namer
  serial_driver
  io_context
)

#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package(
  INSTALL_TO_SHARE
    launch
    params
)
