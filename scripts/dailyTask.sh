#!/bin/bash

the_day=`date -d "$date -2 days" +"%Y-%m-%d"`
featurePath="/home/tuannm/target/demography/data"
workspacePath="/home/phongdk/workspace/customTargeting"
outputPath="/home/phongdk/data_custom_targeting"

echo ${the_day}

export PATH="/home/phongdk/anaconda3/envs/demographic-ml/bin:$PATH"

mainDemographicFile="/home/phongdk/data_ads_targeting/prediction/${the_day}_prob.csv.gz"


cd ${workspacePath}

. scripts/utils.sh 			# load all utility functions
python src/python/main/run_campaigns.py -d ${the_day} -w ${workspacePath} -f ${featurePath} -o ${outputPath}

# while [[ 1 ]]; do
# 	if [ -f "${mainDemographicFile}" ]; 		# waiting for main demographic part finish, then run custom Targeting
# 	then

# 	else
# 		echo "--------------Waiting for ${mainDemographicFile} finish -------"
# 		# sleep 20m
# 	fi
# done
