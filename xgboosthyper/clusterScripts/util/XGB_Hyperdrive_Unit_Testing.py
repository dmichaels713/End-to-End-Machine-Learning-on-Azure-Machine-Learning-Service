# Load in libaries
import unittest
import os

# Load in functions from your other .py files to unit test
from XGB_Hyperdrive_Shared_Functions import create_dict, set_tags

print(os.getcwd())

class TestFunctions(unittest.TestCase):
      
    def setUp(self):
        pass
  
    # Returns True if the output is a dictionary and matches the correct format
    def test_create_dict(self):
        self.assertEqual(create_dict(['Key1', 'Key2'], [123, 456]), {'Key1': 123, 'Key2': 456})
  
    # Returns True if the output is a dictionary and matches the correct format
    def test_set_tags(self):        
        self.assertEqual(['Project', 'Message'], {'Project': 'Test', 'Message': 'Testing'})
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    print('Script Finished')
