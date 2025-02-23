#!/bin/bash

source ~/mediapipe_project/mp_env/bin/activate

# Stop any running VLC instances
pkill vlc

# Print Raspberry Pi's hostname and IP address
echo "Hostname: $(hostname)"
echo "IP Address: $(hostname -I)"

# 1) Start the conversation code in the background, no GUI:
python3 /home/comrade/AI_Conversation_TCPIP.py &

# 2) Start the Eyes code in the foreground with GUI:
python3 /home/comrade/Eyes_TCPIP.py
