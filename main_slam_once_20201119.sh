#!/bin/bash

python3 ~/Git/pyslam_dataprocessing/LINEnotify.py --message start_trials
while :; do
	# timeout -s KILL 75m python3 main_slam_once_1.py
	timeout -s KILL 90m python3 main_slam_once_20201119.py
	EXITVALUE=$?
	printf "\n\n\nEXITVALUE: %s\n\n\n" $EXITVALUE
	python3 ~/Git/pyslam_dataprocessing/LINEnotify.py --message "trial end EXITVALUE: $EXITVALUE"
	if [ $EXITVALUE -eq 0 ]; then
		break
	fi
	if [ $EXITVALUE -eq 4 ]; then # ctrl + C
		break
	fi
done
python3 ~/Git/pyslam_dataprocessing/LINEnotify.py --message finish_trials
# while :; do
# 	python3 main_slam_once.py
# 	echo "EXITVALUE: $?"
# 	if [ $? -eq 0 ]; then
# 		break
# 	else
# 		echo "retry python"
# 	fi
# done

#$ rsync -av /mnt/disk01/GitHub/pyslam_ex/result/TUM_1000/ /home/rkojima/Git/pyslam_dataprocessing/result/TUM/ --exclude "*/frameimage/"
