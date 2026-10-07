import numpy as np
import matplotlib.pyplot as plt 

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.tree import DecisionTreeClassifier 

def load_data(): 
    with open("Decision Tree-Dataset/real.txt", "r", encoding="utf-8") as file:
        real_headlines = [line.strip() for line in file if line.strip()]

    with open("Decision Tree-Dataset/fake.txt", "r", encoding="utf-8") as file:
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




def select_model(X_train, y_train, X_validation, y_validation, X_test, y_test):


    depths = [2, 5, 10, 20, 50, 100, 150, 200, 300]
    validation_accuracies = []


    for depth in depths:
        model = DecisionTreeClassifier(criterion="entropy", max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)
        accuracy = np.mean(predictions == y_validation)

        validation_accuracies.append(accuracy)
        print("max_depth =", depth, "validation accuracy:", accuracy)


    best_index = np.argmax(validation_accuracies)
    best_depth = depths[best_index]

    best_model = DecisionTreeClassifier(criterion="entropy", max_depth=best_depth, random_state=42)
    best_model.fit(X_train, y_train)

    test_predictions = best_model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test)

    print()
    print("Best max_depth:", best_depth)
    print("Actual depth of the trained tree:", best_model.get_depth())
    print("Test accuracy:", test_accuracy)



    plt.plot(depths, validation_accuracies, marker="o")
    plt.xlabel("max_depth")
    plt.ylabel("Validation accuracy")
    plt.title("Validation accuracy vs. max_depth")
    plt.grid(True)
    plt.savefig("validation_accuracy_vs_max_depth.png")
    plt.show()


    return best_model




if __name__ == "__main__":
    X_train, y_train, X_validation, y_validation, X_test, y_test, vectorizer = load_data()
    select_model(X_train, y_train, X_validation, y_validation, X_test, y_test)