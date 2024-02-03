from typing import  Any, List , Optional , Dict
from pydantic import BaseModel
from bikeshare_model.processing.validation import DataInputSchema
class PredictionResults(BaseModel):
    errors : Optional[Dict]
    version : str
    predictions: List[float]


class MultipleDataInputs(BaseModel):
    inputs : List[DataInputSchema]

    class Config:
        schema_extra ={
            "example":{
                "inputs":[
                    {
                        'dteday': '2012-11-05',
                        'season': 'winter',
                        'hr': '6am',
                        'holiday': 'No',
                        'weekday': 'Mon',
                        'workingday': 'Yes',
                        'weathersit': 'Mist',
                        'temp': 6.1,
                        'atemp': 3.0014000000000003,
                        'hum': 49.0,
                        'windspeed': 19.0012,
                        'casual': 4,
                        'registered': 135,
                        "yr": 2012,
                        "mnth": "May"
                    }
                ]
            }
        }

