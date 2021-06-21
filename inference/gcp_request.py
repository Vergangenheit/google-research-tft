import googleapiclient.discovery
from googleapiclient.discovery import Resource
import pandas as pd
from pandas import DataFrame
from typing import Dict, List
try:
    from dotenv import load_dotenv

    load_dotenv()
except:
    print('No ".env" file or python-dotenv not installed... Using default env variables...')


MODEL_NAME: str = "tft_sorgenia_logs"
PROJECT_ID = "omnienergy-316210"
sample_path: str = r'C:\Users\Lorenzo\PycharmProjects\TFT\outputs\data\sorgenia_wind\data\sorgenia_wind\data\sorgenia_wind_inference_sample.csv'

instances: List = pd.read_csv(sample_path).values.tolist()
service: Resource = googleapiclient.discovery.build('ml', 'v1')
name: str = 'projects/{}/models/{}/versions/{}'.format(PROJECT_ID, MODEL_NAME, 'v2')

response: Dict = service.projects().predict(
    name=name,
    body={'instances': instances}
).execute()

if 'error' in response:
    raise RuntimeError(response['error'])
else:
    print(response['predictions'])

