import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

import os 

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','proprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        sex: str,
        smoker: str,
        region: str,
        age: int,
        bmi: float,
        children: int,
        ):

        self.sex = sex

        self.smoker = smoker

        self.region = region

        self.age = age

        self.bmi = bmi

        self.children = children    
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "sex": [self.sex],
                "smoker": [self.smoker],
                "region": [self.region],
                "age": [self.age],
                "bmi": [self.bmi],
                "children": [self.children],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)