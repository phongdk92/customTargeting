#!/bin/bash

# export PATH="/usr/local/envs/demographic-ml/bin/python:$PATH"
#source activate daskEnv
source activate demographic-ml
which python

date=$1
workspacePath=$2
featurePath=$3
outputPath=$4
cateID=$5
hyperParams=$6
newLabels=$7
metric=$8
cateID2=$9 # optinal for more inventory

cd ${workspacePath}


fileTrain="${featurePath}/user_features_out_training_${date}.gz"
fileTest="${featurePath}/user_features_out_testing_${date}.gz"
fileAllUsers="${featurePath}/user_features_out_demographic_${date}.gz"
#fileAllUsers="${featurePath}/user_features_out_testing_${date}.gz"

outputPrediction="${outputPath}/prediction"
outputFile="${outputPrediction}/${date}.gz"

modelByWeek="${outputPath}/model_by_week"
outputModel="${modelByWeek}/${date}"

mkdir -p ${outputPrediction}
mkdir -p ${modelByWeek}


function demography_train {
#	echo "src/python/main/trainer_dask.py -f ${fileTrain} -q ${fileTest} -d ${outputModel} -hp ${hyperParams} -t -hdfs"
#	python src/python/main/trainer_dask.py -f ${fileTrain} -q ${fileTest} -d ${outputModel} -hp ${hyperParams} -t -hdfs
    echo "src/python/main/trainer.py -f ${fileTrain} -q ${fileTest} -d ${outputModel} -nl ${newLabels} -hp ${hyperParams} -t -me ${metric}"
	python src/python/main/trainer.py -f ${fileTrain} -q ${fileTest} -d ${outputModel} -nl ${newLabels} -hp ${hyperParams} -t -me ${metric}
	checkResult_slack $? "[Custom category ${cateID} ]-------Training procedure"

}

if [[ -f "${fileTrain}" ]] && [[ -f "${fileTest}" ]]; then       # train new model
    demography_train
fi

function demography_predict {
#	echo "src/python/main/tester_dask.py -md ${modelByWeek} -q ${fileAllUsers} -o ${outputFile} -hdfs"
#	python src/python/main/tester_dask.py -md ${modelByWeek} -q ${fileAllUsers} -o ${outputFile} -hdfs
    echo "src/python/main/predictor.py -md ${modelByWeek} -q ${fileAllUsers} -o ${outputFile} -cs 500000 -cid ${cateID} -cid2 ${cateID2}"
	python src/python/main/predictor.py -md ${modelByWeek} -q ${fileAllUsers} -o ${outputFile} -cs 500000 -cid ${cateID} -cid2 ${cateID2}
	checkResult_slack $? "[Custom category ${cateID}]-------Prediction procedure"

}

if [[ -f "${fileAllUsers}" ]]; then
    demography_predict
else
    echo "File ${fileAllUsers} not found : "
fi
