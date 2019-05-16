## This a custom targeting project based on Sale Requirements

<!---
python src/python/main/trainer_dask.py -f ~tuannm/target/demography/data/user_features_out_training_2019-04-08.gz -q ~tuannm/target/demography/data/user_features_out_testing_2019-04-08.gz -nl external_data/new_age_label_22+.csv -d ~/tmp/2019-04-08 -hp best_params/topica_22+_305422 -t


python src/python/main/predictor_dask.py -q ~tuannm/target/demography/data/user_features_out_testing_2019-04-08.gz -md ~/tmp/model_by_week -cid 11111 -o ~/tmp/test_data.gz -cs 10000


python src/python/main/trainer_dask.py -f hdfs:/user/phongdk/target/demography/data/user_features_out_training_2019-04-15.gz -q hdfs:/user/phongdk/target/demography/data/user_features_out_testing_2019-04-15.gz -nl external_data/new_age_label_22+.csv -d data_custom_targeting/topica_22_305422/model_by_week/2019-04-15 -hp best_params/topica_22+_305422 -t -hdfs


python src/python/main/predictor_dask.py -q hdfs:/user/phongdk/target/demography/data/user_features_out_testing_2019-04-15.gz -md data_custom_targeting/topica_22_305422/model_by_week -cid 11111 -o hdfs:/user/phongdk/data_custom_targeting/topica_22_305422/prediction/test_data.gz -l ~/log_dask -hdfs
--->

### 1. To Install Environment
    conda env create -n demographic-ml -f environment/enviroment.yml
    
### 2. Directory
    - src: source code
    - scripts: script to run each campaign
    - environment: contain a file *.yml to build compatiable environment
    - external_data: contain a file in format browser_id DATE_OF_BIRTH/YEAR_OF_BIRTH
    
### 3. Run bash script daily.
```bash
bash scripts/dailyTask.sh
``` 
    This script will call run_campaigns.py file in order to do the following work:
        - Loop over all campaigns
        - If a campaign is choosen to run:
            - Create new label for data
            - Generate new training/testing data
            - Predict unlabelled data
        - Join results of all campaigns into a single file.
           
    The campaign information is stored in json format like 
    {
        "name": "topica#554027",
        "start_date": "2019-04-10",
        "end_date": "2019-05-27",
        "is_runnable": true,
        "cateID": "15517",
        "metric": "precision",
        "cateID2": "0",
        "low": "22",
        "high": "100"
    }

For each campaign, the training procedure (one per week) is
```bash
python src/python/main/trainer.py   -f ${fileTrain} \   # path to file train
                                    -q ${fileTest}  \   # path to file test
                                    -d ${outputModel} \ # path to output model
                                    -nl ${newLabels}  \ # path to new label file 
                                    -hp ${hyperParams} \# path to hyperparameters
                                    -me ${metric} \     # metric to get optimal threshold (f1_score, precision)
                                    -t \                # is_train model (default = False)
                                    -op                 # is optimize parameter (default = False)
	
```
The `-op` parameter is only used for optimizing parameters one time only since it's so time-consuming (1-2 days to run).

Anh the prediction procedure (daily) is
```bash
python src/python/main/predictor.py -md ${modelByWeek} \    # path to trained model
                                    -q ${fileAllUsers} \    # path to file to predict
                                    -o ${outputFile} \      # path to output file
                                    -cs 500000 \            # chunk size    
                                    -cid ${cateID} \        # cateID (optimal threshold is used for this Cate)
                                    -cid2 ${cateID2}        # cateID2 (if need more inventory)
```