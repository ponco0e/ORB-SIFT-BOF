include(CXX11.cmake)


project(cv_reconocimientoEscenas CXX)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
find_package( OpenCV REQUIRED )
add_executable(${PROJECT_NAME} ${SRC_LIST})

include_directories("${PROJECT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
TARGET_LINK_LIBRARIES(${PROJECT_NAME}  "${CMAKE_CURRENT_BINARY_DIR}/libvl.so")
enable_cxx11()
