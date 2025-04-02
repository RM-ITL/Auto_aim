gnome-terminal --title="serial_red" --working-directory=$PWD -- bash -c \
 ". install/setup.bash && \
 ros2 run pkg02_helloworld_py serial_send; \
  exec bash" 
sleep 10
gnome-terminal --title="red" --working-directory=$PWD -- bash -c \
 ". install/setup.bash && \
 ros2 launch launch_node launch_red.launch.py; \
  exec bash"
 

