import joblib
import pandas as pd

class LinearRegression:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.model = joblib.load(path_to_artifacts + "linear_reg.joblib")
    
    def preprocessing(self, input_data):
        input_data = pd.DataFrame(input_data, index=[0])
        return input_data
        
    def predict(self, input_data):
        return self.model.predict_proba(input_data)

    def postprocessing(self, input_data):
        output = input_data[32]
        return {"prediction": output, "status": "OK"}
    
    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction
