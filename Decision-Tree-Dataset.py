import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.tree import DecisionTreeClassifier 

def load_data(): 
    with open("real.txt", "r", encoding="utf-8") as file: 
        real_headlines = [line.strip() for line in file if line.strip()] 
        
