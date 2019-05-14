
#!/bin/bash
USERNAME=ubuntu

#LAN->1,2,3
IP1=54.202.58.29		#Oregon Sameer 
IP2=54.214.106.239		#Oregon Anonymous
IP3=34.223.225.30		#Oregon John

#WAN->1,2,4
IP4=35.166.8.86			#Ohio


#########################################################################################
NETWORK=SecureML		# NETWORK {SecureML, Sarda, MiniONN, LeNet, AlexNet, and VGG16}
DATASET=MNIST 			# DATASET {MNIST, CIFAR10, and ImageNet}
SECURITY=Semi-honest	# SECURITY {Semi-honest or Malicious} 
RUN_TYPE=LAN 			# RUN_TYPE {LAN or WAN or localhost}
PRINT_TO_FILE=true		# PRINT_TO_FILE {true or false}
FILENAME=time.txt
#########################################################################################

if [[ $PRINT_TO_FILE = true ]]; then
	printf "%s\n" "---------------------------------------------" >> $FILENAME
	printf "%s %s %s %s\n" $RUN_TYPE $NETWORK $DATASET $SECURITY >> $FILENAME
	printf "%s\n" "---------------------------------------------" >> $FILENAME
fi


if [[ $RUN_TYPE = LAN ]]; then
	ssh -i ~/.ssh/ccs_oregon_snwagh.pem $USERNAME@$IP1 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 0 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt; less time.txt" & 
	ssh -i ~/.ssh/ccs_oregon_anonymous.pem $USERNAME@$IP2 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 1 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt" & 
	ssh -i ~/.ssh/ccs_oregon_john.pem $USERNAME@$IP3 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 2 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt" & 
elif [[ $RUN_TYPE = WAN ]]; then
	ssh -i ~/.ssh/ccs_oregon_snwagh.pem $USERNAME@$IP1 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 0 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt; less time.txt" & 
	ssh -i ~/.ssh/ccs_oregon_anonymous.pem $USERNAME@$IP2 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 1 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt" & 
	ssh -i ~/.ssh/ccs_oregon_john.pem $USERNAME@$IP4 "pkill BMRPassive.out; echo clean completed; cd malicious-security; make; chmod +x BMRPassive.out; ./BMRPassive.out 2 files/IP_$RUN_TYPE files/keyA files/keyAB files/keyAC $NETWORK $DATASET $SECURITY 1>./time.txt" & 
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



#CLEAN, BUILD AND RUN
#LAN
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME1@$IP1 "pkill BMRPassive.out; echo clean completed; cd SecureDNN; chmod +x run; git pull; make clean; make; echo compile done; ./run; less output/time0.txt" & 
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME2@$IP2 "pkill BMRPassive.out; echo clean completed; cd SecureDNN; chmod +x run; git pull; make clean; make; echo compile done; ./run" & 
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME3@$IP3 "pkill BMRPassive.out; echo clean completed; cd SecureDNN; chmod +x run; git pull; make clean; make; echo compile done; ./run" & 
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME4@$IP4 "pkill BMRPassive.out; echo clean completed; cd SecureDNN; chmod +x run; git pull; make clean; make; echo compile done; ./run" &
#WAN
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME1@$IP1 "pkill BMRPassive.out; echo clean completed; cd SecureDNN; chmod +x run; git pull; make clean; make; echo compile done; ./run; less output/time0.txt" & 
# ssh -i ~/.ssh/msr_cnn_va.pem $USERNAME5@$IP5 "pkill BMRPassive.out; echo clean completed; cd SecureDNN; chmod +x run; git pull; make clean; make; echo compile done; ./run" & 
# ssh -i ~/.ssh/msr_cnn_west.pem $USERNAME6@$IP6 "pkill BMRPassive.out; echo clean completed; cd SecureDNN; chmod +x run; git pull; make clean; make; echo compile done; ./run" &
# ssh -i ~/.ssh/msr_cnn_ca.pem $USERNAME7@$IP7 "pkill BMRPassive.out; echo clean completed; cd SecureDNN; chmod +x run; git pull; make clean; make; echo compile done; ./run" & 




########################################## OTHERS ##########################################
#GENERIC COMMANDS
#git config --global credential.helper store; git pull
#sudo apt-get update; sudo apt-get install g++; sudo apt-get install libssl-dev; sudo apt install make 
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME1@$IP1 "cd SecureDNN; git pull; cp azure_exec run"
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME2@$IP2 "cd SecureDNN; git pull; cp azure_exec run"
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME3@$IP3 "cd SecureDNN; git pull; cp azure_exec run"
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME4@$IP4 "cd SecureDNN; git pull; cp azure_exec run"
# ssh -i ~/.ssh/msr_cnn_va.pem $USERNAME5@$IP5 "cd SecureDNN; git pull; cp azure_exec run"
# ssh -i ~/.ssh/msr_cnn_west.pem $USERNAME6@$IP6 "cd SecureDNN; git pull; cp azure_exec run"
# ssh -i ~/.ssh/msr_cnn_ca.pem $USERNAME7@$IP7 "cd SecureDNN; git pull; cp azure_exec run"

# ssh -i ~/.ssh/msr_cnn.pem $USERNAME1@$IP1 
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME2@$IP2 
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME3@$IP3 
# ssh -i ~/.ssh/msr_cnn.pem $USERNAME4@$IP4 
# ssh -i ~/.ssh/msr_cnn_va.pem $USERNAME5@$IP5 
# ssh -i ~/.ssh/msr_cnn_west.pem $USERNAME6@$IP6 
# ssh -i ~/.ssh/msr_cnn_ca.pem $USERNAME7@$IP7 


