import unittest
import numpy as np

class TestQTable(unittest.TestCase):
    def __init__ (self, *args, **kwargs):
        super(TestQTable, self).__init__(*args, **kwargs)
        self.q_table = np.load("q_table.npy")
        
    def test_q_table_shape(self):
        print(self.q_table)
        # Check if the Q-table has the expected shape
        expected_shape = (3, 3)  # Example shape, adjust as necessary
        self.assertEqual(self.q_table.shape, expected_shape)

    def test_q_table_sum(self):
        # Check if the sum of Q-values for each state is within a reasonable range
        for state in range(self.q_table.shape[0]):
            state_sum = np.sum(self.q_table[state])
            self.assertTrue(0 <= state_sum <= 3, f"Q-values for state {state} do not sum to 1")
    
if __name__ == "__main__":
    unittest.main()