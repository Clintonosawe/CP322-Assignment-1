import numpy as np
import matplotlib.pyplot as plt 

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.tree import DecisionTreeClassifier 

def load_data(): 
    with open("real.txt", "r", encoding="utf-8") as file: 
        real_headlines = [line.strip() for line in file if line.strip()] 

    with open("fake.txt", "r", encoding="utf-8")  as file:
        fake_headlines = [line.strip() for line in file if line.strip()]

    headlines = real_headlines + fake_headlines 

    labels = np.array( 
        [1] * len(real_headlines) + 
        [0] * len(fake_headlines) 
    )

   # Convert text into numerical word-count features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(headlines)

    # Randomly shuffle the dataset
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(headlines))

    X = X[indices]
    labels = labels[indices]

    # 70% training, 15% validation, 15% test
    n = len(labels)
    train_end = int(0.70 * n)
    validation_end = int(0.85 * n)

    X_train = X[:train_end]
    y_train = labels[:train_end]

    X_validation = X[train_end:validation_end]
    y_validation = labels[train_end:validation_end]

    X_test = X[validation_end:]
    y_test = labels[validation_end:]

    print("Training examples:", len(y_train))
    print("Validation examples:", len(y_validation))
    print("Test examples:", len(y_test))

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        vectorizer
    )

if __name__ == "__main__":
    data = load_data()