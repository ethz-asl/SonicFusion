cmake_minimum_required(VERSION 3.8)
project(sonic_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

# CPP stuff if needed ...
# https://roboticsbackend.com/ros2-package-for-both-python-and-cpp-nodes/

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/sonic_trajectory_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install Launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Make scripts executable to use --symlink-install
execute_process(COMMAND sh -c "chmod +x ${CMAKE_CURRENT_LIST_DIR}/scripts/*.py")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
