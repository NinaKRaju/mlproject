import joblib
import pandas as pd

class RandomForest:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.encoders = joblib.load(path_to_artifacts + "encoders_lin2.joblib")
        self.model = joblib.load(path_to_artifacts + "studentrf.joblib")

    def preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        # convert categoricals
        for column in [
            "school",
            "sex",
            "address",
            "famsize",
            "Pstatus",
            "Mjob",
            "Fjob",
            "reason",
            "guardian",
            "schoolsup",
            "famsup",
            "paid",
            "activities",
            "nursery",
            "higher",
            "internet",
            "romantic",
        ]:
            categorical_convert = self.encoders[column]
            input_data[column] = categorical_convert.transform(input_data[column])

        return input_data

    def predict(self, input_data):
        return self.model.predict(input_data)
        

    def postprocessing(self, input_data):
        label = "fail"
        if input_data > 13:
            label = "pass"
        return {"prediction": input_data, "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e), "prediction": self.predict(input_data)[0]}

        return prediction
