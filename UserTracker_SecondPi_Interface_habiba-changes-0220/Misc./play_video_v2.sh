#!/bin/bash

source ~/mediapipe_project/mp_env/bin/activate

# Stop any running VLC instances
pkill vlc

# Print Raspberry Pi's hostname and IP address
echo "Hostname: $(hostname)"
echo "IP Address: $(hostname -I)"

# Run the Python script
python3 /home/comrade/Eyes_TCPIP.py
