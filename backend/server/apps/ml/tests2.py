from django.test import TestCase

from apps.ml.rain_classifier.random_rainforest import RandomForestClassifier
import inspect
from apps.ml.registry import MLRegistry

class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = {
        
            "pressure": 1025.9,
            "maxtemp": 19.9,
            "temperature": 18.3,
            "mintemp": 16.8,
            "dewpoint": 13.1,
            "humidity": 72,
            "cloud": 49,
            "sunshine": 9.3,
            "winddirection": 80,
            "windspeed": 26.3,
        }
        my_alg = RandomForestClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('yes', response['label'])
        
