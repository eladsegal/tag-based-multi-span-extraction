import os
#print(os.environ.get("PATH")) # Run it when the environment is activated and use the related env values to overwrite the PATH variable 
os.environ['PATH'] = 'C:\\Users\\Elad\\Anaconda3\\envs\\allennlp;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Library\\mingw-w64\\bin;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Library\\usr\\bin;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Library\\bin;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\Scripts;C:\\Users\\Elad\\Anaconda3\\envs\\allennlp\\bin;C:\\Users\\Elad\\Anaconda3\\condabin;'

import unittest
from src.custom_drop_em_and_f1 import CustomDropEmAndF1

class CustomDropMetricTest(unittest.TestCase):

    def testStandard(self):
        drop_metrics = CustomDropEmAndF1()
        prediction = ['good']
        score, maximizing_ground_truth = drop_metrics.call(prediction, [{'spans': ['not good']}, {'spans': ['good']}], 'some ability')
        self.assertEqual(tuple([1.0, 1.0]), score)
        self.assertEqual(tuple(prediction), maximizing_ground_truth)

if __name__ == '__main__':
    unittest.main()