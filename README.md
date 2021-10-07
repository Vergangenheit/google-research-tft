# google-research-tft

Putting the awesome code from https://github.com/google-research/google-research/tree/master/tft in a separate standalone repository to dig through it.

**THIS REPO IS MAINLY FOR TRAINING PURPOSES**

The model is then deployed in https://gitlab.com/omnienergysrl/ml-data-science/production-prediction-vulcanodon 

**INFERENCE-PREDICTION**:

The main function in script_train_fixed_params.py is configured to run the prediction on the testset. It does that by generating two csv files in the project folder plus the target dataframe if requested.

Example:

./outputs/fixed

├── checkpoint

├── **p50.csv**

├── **p90.csv**

├── params.csv

├── results.csv

├── sorgenia_wind_ckpt.hdf5

├── sorgenia_wind_no_forecasts_ckpt.hdf5

├── **targets.csv**

├── TemporalFusionTransformer.ckpt.data-00000-of-00001

├── TemporalFusionTransformer.ckpt.index

├── TemporalFusionTransformer.ckpt.meta

└── tmp/

