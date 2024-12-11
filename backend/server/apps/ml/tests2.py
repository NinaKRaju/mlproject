from django.test import TestCase

from apps.ml.student_classifier.linear_reg import LinearRegression
import inspect
from apps.ml.registry import MLRegistry

class MLTests(TestCase):
    def test_linreg_algorithm(self):
        input_data = {
            "school": "GP",
            "sex": "F",
            "age": 18,
            "address": "U",
            "famsize": "GT3",
            "Pstatus": "A",
            "Medu": 4,
            "Fedu": 4,
            "Mjob": "at_home",
            "Fjob": "teacher",
            "reason": "course",
            "guardian": "mother",
            "traveltime": 2,
            "studytime": 2,
            "failures": 0,
            "schoolsup": "yes",
            "famsup": "no",
            "paid": "no",
            "activities": "no",
            "nursery": "yes",
            "higher": "yes",
            "internet": "no",
            "romantic": "no",
            "famrel": 4,
            "freetime": 3,
            "goout": 4,
            "Dalc": 1,
            "Walc": 1,
            "health": 3,
            "absences": 6,
            "G1": 15,
            "G2": 16,
    
        }
        my_alg = LinearRegression()
        response = my_alg.compute_prediction(input_data)
        print(response)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('pass', response['label'])

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "student_classifier"
        algorithm_object = LinearRegression()
        algorithm_name = "linear regression"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Piotr"
        algorithm_description = "Linear Regression with super impressive pre- and post-processing"
        algorithm_code = inspect.getsource(LinearRegression)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
