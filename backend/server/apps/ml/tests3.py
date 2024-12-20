from django.test import TestCase

from apps.ml.student_classifier.random_forest2 import RandomForest
import inspect

class MLTests(TestCase):
    def test_rf_algorithm(self):
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
            "G1": 5,
            "G2": 6,
    
        }
        
        my_alg = RandomForest()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('fail', response['label'])
        
