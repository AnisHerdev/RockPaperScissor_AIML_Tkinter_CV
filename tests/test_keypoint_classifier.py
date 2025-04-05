import unittest
from model import KeyPointClassifier

class TestKeyPointClassifier(unittest.TestCase):
    def test_keypoint_classifier(self):
        classifier = KeyPointClassifier()
        sample_input = [0.1] * 42  # Example normalized input
        result = classifier(sample_input)
        self.assertIn(result, [0, 1, 2, 3])  # Ensure result is a valid class ID

if __name__ == "__main__":
    unittest.main()