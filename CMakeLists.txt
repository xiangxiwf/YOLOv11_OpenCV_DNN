cmake_minimum_required(VERSION 3.10)
project(YOLOv11_Detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

add_executable(YOLOv11_detector
    main.cpp
    YOLOv11.cpp
    YOLOv11.h
)

target_link_libraries(YOLOv11_detector ${OpenCV_LIBS})

set_target_properties(YOLOv11_detector PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
