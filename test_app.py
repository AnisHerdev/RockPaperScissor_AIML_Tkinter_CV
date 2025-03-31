import unittest
from app import main_function, helper_function

class TestApp(unittest.TestCase):
    def test_main_function(self):
        # Arrange
        input_data = {"gesture": "wave"}  # Example input for gesture recognition
        expected_output = "Gesture recognized: wave"  # Expected output based on input
        
        # Act
        result = main_function(input_data)
        
        # Assert
        self.assertEqual(result, expected_output)

    def test_helper_function(self):
        # Arrange
        input_data = [1, 2, 3, 4, 5]  # Example input for a helper function
        expected_output = 15  # Expected output (e.g., sum of the list)
        
        # Act
        result = helper_function(input_data)
        
        # Assert
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()
