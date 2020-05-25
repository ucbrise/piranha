
#!/bin/bash
USERNAME=ubuntu

#LAN->1,2,3
IP1=54.189.9.223		#Oregon Sameer 
IP2=18.236.118.29		#Oregon Sameer
IP3=54.202.116.205		#Oregon Sameer

#WAN->1,4,5
IP4=54.202.116.205		#Oregon
IP5=3.22.241.177		#Ohio


#########################################################################################
NETWORK=AlexNet # NETWORK {SecureML, Sarda, MiniONN, LeNet, AlexNet, and VGG16}
DATASET=CIFAR10 # DATASET {MNIST, CIFAR10, and ImageNet}
SECURITY=Semi-honest # SECURITY {Semi-honest or Malicious} 
RUN_TYPE=localhost # RUN_TYPE {LAN or WAN or localhost}
PRINT_TO_FILE=false		# PRINT_TO_FILE {true or false}
FILENAME=time.txt
#########################################################################################

if [[ $PRINT_TO_FILE = true ]]; then
	printf "%s\n" "---------------------------------------------" >> $FILENAME
	printf "%s %s %s %s\n" $RUN_TYPE $NETWORK $DATASET $SECURITY >> $FILENAME
	printf "%s\n" "---------------------------------------------" >> $FILENAME
fi


if [[ $RUN_TYPE = LAN ]]; then
	ssh -i ~/.ssh/falcon_sp_oregon.pem $USERNAME@$IP1 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 0 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt; less time.txt" & 
	ssh -i ~/.ssh/falcon_sp_oregon.pem $USERNAME@$IP2 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 1 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt" & 
	ssh -i ~/.ssh/falcon_sp_oregon.pem $USERNAME@$IP3 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 2 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt" & 
elif [[ $RUN_TYPE = WAN ]]; then
	ssh -i ~/.ssh/falcon_sp_oregon.pem $USERNAME@$IP1 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 0 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt; less time.txt" & 
	ssh -i ~/.ssh/falcon_sp_oregon.pem $USERNAME@$IP4 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 1 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt" & 
	ssh -i ~/.ssh/falcon_sp_ohio.pem $USERNAME@$IP5 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 2 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt" & 
elif [[ $RUN_TYPE = localhost ]]; then
	make
	./BMRPassive.out 1 files/IP_$RUN_TYPE files/keyB files/keyBC files/keyAB $NETWORK $DATASET $SECURITY >/dev/null &
	./BMRPassive.out 2 files/IP_$RUN_TYPE files/keyC files/keyAC files/keyBC $NETWORK $DATASET $SECURITY >/dev/null &
	if [[ $PRINT_TO_FILE = true ]]; then
		./BMRPassive.out 0 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY >> $FILENAME
	else
		./BMRPassive.out 0 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 
	fi
else
	echo "RUN_TYPE error" 
fi




########################################## SET-UP COMMANDS ##########################################
#sudo apt-get update; sudo apt-get install g++; sudo apt-get install libssl-dev; sudo apt install make; sudo apt-get install iperf3 
#git clone https://github.com/snwagh/malicious-security.git; cd malicious-security; git config credential.helper cache

# ssh -i ~/.ssh/falcon_sp_oregon.pem ubuntu@18.237.39.209
# ssh -i ~/.ssh/falcon_sp_oregon.pem ubuntu@34.221.35.166
# ssh -i ~/.ssh/falcon_sp_oregon.pem ubuntu@34.219.97.126
