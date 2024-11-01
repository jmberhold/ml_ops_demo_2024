# test_model.py
import unittest
from joblib import load
import os

class ModelTest(unittest.TestCase):
    def test_model_exists(self):
        # Test if the model file exists
        self.assertTrue(os.path.exists('model/model.pkl'), "Model file not found!")

    def test_model_prediction(self):
        # Test if model can make predictions
        model = load('model/model.pkl')
        sample_data = [[5.1, 3.5, 1.4, 0.2]]  # Iris dataset example
        prediction = model.predict(sample_data)
        self.assertIn(prediction[0], [0, 1, 2], "Unexpected prediction value!")

if __name__ == '__main__':
    unittest.main()