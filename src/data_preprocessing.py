import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder ,StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os


from src.logger import get_logger
from src.custom_exception import CustomExcepion


logger = get_logger(__name__)

class DataProcessing:
    def __init__(self,input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.features = None

        os.makedirs(self.output_path,exist_ok=True)

        logger.info("Data processing initialised")

    def load_data(self):

        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("Data loaded Sucesfully")
        
        except Exception as e:
            logger.error(f"Error while loading the data {e}")
            raise CustomExcepion("Faild to load the data",e)
        
    def preprocess(self):

        try:
            self.df = self.df.fillna(self.df.mean(numeric_only=True))
        

            columns_to_encode = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            label_encoder = LabelEncoder()
            for col in columns_to_encode:
                self.df[col] = label_encoder.fit_transform(self.df[col])
            
            logger.info("All the basic data prepocessing is done")
        
        except Exception as e:
            logger.error(f"Error while prepossing the data {e}")
            raise CustomExcepion("Faild to prepossing the data",e)
        
    
    def split_scale_save(self):
        try:

            X = self.df.drop(["id","stroke"], axis=1)
            y = self.df["stroke"]

            self.features = X.columns

            X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.2 , random_state=42 , stratify=y)
            
            
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            X_train = pd.DataFrame(X_train, columns=X.columns)
            X_test = pd.DataFrame(X_test, columns=X.columns)

            


            joblib.dump(X_train, os.path.join(self.output_path,"X_train.pkl"))
            joblib.dump(X_test, os.path.join(self.output_path,"X_test.pkl"))
            joblib.dump(y_train, os.path.join(self.output_path,"y_train.pkl"))
            joblib.dump(y_test, os.path.join(self.output_path,"y_test.pkl"))


            joblib.dump(scaler, os.path.join(self.output_path,"scaler.pkl"))
            
            logger.info("All files saved sucesfully for data processing")
        
        except Exception as e:
            logger.error(f"Error while split scale and save the data {e}")
            raise CustomExcepion("Faild to split scale and save the data",e)
        
    
    def run(self):
        self.load_data()
        self.preprocess()
        self.split_scale_save()


if __name__ == "__main__":
    prepocessing = DataProcessing("artifacts/raw/data.csv","artifacts/processed")
    prepocessing.run()