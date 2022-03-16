from datasets import datasets
import numpy as np
import pickle
size = 100
dataset = datasets()
# ds = dataset.create("C:/Users/nico/Dropbox/Coursera/DeepLearning/ml_aplicado/datasets/pikachu", (size,size))
ds = dataset.create("datasets/pikachu", (size,size))
pickle.dump(ds, open("aer.pickle","wb"))