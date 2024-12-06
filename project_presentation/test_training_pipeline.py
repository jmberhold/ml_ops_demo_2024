import os
import unittest
import json
import sys

class TestTrainingPipeline(unittest.TestCase):

    def setUp(self):
        """Set up any required variables or paths."""
        self.data_location = sys.argv[1]  # Path where data is expected to be saved
        self.model_directory = sys.argv[2]  # Directory where models are saved
        self.model_log = sys.argv[2]  # Path to the JSON log file

    def test_data_saved_in_data_location(self):
        """Test to verify that the data exists in the data location."""
        # Ensure the data directory exists
        self.assertTrue(
            os.path.exists(self.data_location),
            f"Data location '{self.data_location}' does not exist."
        )

        # Check that the data directory is not empty
        self.assertTrue(
            os.listdir(self.data_location),
            f"Data location '{self.data_location}' is empty."
        )

    def test_json_update_and_model_location(self):
        """Test to check if the JSON log file is updated and model exists."""
        # Ensure the JSON log file exists
        self.assertTrue(
            os.path.exists(self.model_log),
            f"Log file '{self.model_log}' does not exist."
        )

        # Load the JSON log
        with open(self.model_log, "r") as file:
            logs = json.load(file)

        # Ensure the log is not empty
        self.assertTrue(logs, "JSON log file is empty.")

        # Get the latest model from the JSON log
        # (timestamps are lexicographically sortable)
        latest_model = max(logs, key=lambda x: x["timestamp"])

        # Check if the model path exists
        model_location = latest_model["model_path"]
        model_name = latest_model["model_name"]
        self.assertTrue(
            os.path.exists(os.path.join(model_location, f"model_{model_name}.pt")),
            f"Model '{latest_model['model_name']}' not found at '{model_location}'."
        )

if __name__ == "__main__":
    # Open a file to store the results
    with open("unittest_results.txt", "w") as f:
        # Create a TextTestRunner that writes to the file
        # stream=f ensures test results are written to the file
        # verbosity=2 provides detailed output.
        runner = unittest.TextTestRunner(stream=f, verbosity=2)
        # Run the tests
        unittest.main(testRunner=runner, exit=False)