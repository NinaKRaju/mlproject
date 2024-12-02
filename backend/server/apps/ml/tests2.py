from django.test import TestCase

from apps.ml.rain_classifier.random_rainforest import RandomForestClassifier
import inspect
from apps.ml.registry import MLRegistry

class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = {
            "pressure": 37,
            "maxtemp": "Private",
            "temperature": 34146,
            "mintemp": "HS-grad",
            "dewpoint": 9,
            "humidity": "Married-civ-spouse",
            "cloud": "Craft-repair",
            "sunshine": "Husband",
            "winddirection": "White",
            "windspeed": "Male",
        }
        my_alg = RandomForestClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('<=50K', response['label'])
        
