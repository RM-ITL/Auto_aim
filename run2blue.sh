gnome-terminal --title="serial_blue" --working-directory=$PWD -- bash -c \
 ". install/setup.bash && \
 ros2 run pkg02_helloworld_py serial_send; \
  exec bash" 
sleep 10
gnome-terminal --title="blue" --working-directory=$PWD -- bash -c \
 ". install/setup.bash && \
 ros2 launch launch_node launch_blue.launch.py; \
  exec bash"
