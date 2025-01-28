# logging onto pi 2 (LCD screen pi)

## Notes: 
- I configured the pi's ethernet address to be 192.168.0.3
- It's configured to connect to my home network (Habiba) and the university's UMSecure wifi. 

## Steps: 
1. Connect pi with ethernet to computer
2. ssh into 192.168.0.3 using ssh comrade@192.168.0.3
3. password is comrade
4. once in, disable the video or script autoplayed by writing in the terminal using pkill vlc
5. to connect wirelessly get the ip address that the university assigned to it using hostname -I
6. you will get two ip addresses as such: 192.168.0.3 140.193.235.72 
7. you will notice one of them is not  192.168.0.3
8. Exit the ssh session; and ssh into the other ip address instead as such: ssh comrade@140.193.235.72 

## More notes
- To run the poseDetection1.py program you have to use the virtual envioronment using source ~/mediapipe_project/mp_env/bin/activate
  
