import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
import time

# Load the dataset from the provided file path
def load_data(file_path):
    data = np.loadtxt(file_path)
    return data

# Separate the features and labels from the dataset
def separate_features_labels(data):
    # Features (excluding the first column which is the label)
    X = data[:, 1:]  
    # Labels
    y = data[:, 0].astype(int)  
    return X, y

# Perform k-nearest neighbor classification using Leave-One-Out cross-validation
def nearest_neighbor_loo(X, y, k=1):
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    knn = KNeighborsClassifier(n_neighbors=k)
    
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(X_train, y_train)
        y_pred.append(knn.predict(X_test)[0])
        y_true.append(y_test[0])
    
    return np.array(y_true), np.array(y_pred)

# Calculate the accuracy of the model
def calculate_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

# Forward Selection Algorithm with logging
def forward_selection(X, y, k=1):
    n_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))
    best_accuracy = 0
    best_features = []
    log = []

    print("Beginning search.")

    while remaining_features:
        best_feature = None
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_subset = X[:, current_features]
            y_true, y_pred = nearest_neighbor_loo(X_subset, y, k)
            accuracy = calculate_accuracy(y_true, y_pred)
            print(f"Using feature(s) {set(current_features)} accuracy is {accuracy*100:.1f}%")
            log.append((set(current_features), accuracy))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature
                best_features = current_features
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break

    return best_features, best_accuracy, log

# Backward Elimination Algorithm with logging
def backward_elimination(X, y, k=1):
    n_features = X.shape[1]
    selected_features = list(range(n_features))
    y_true, y_pred = nearest_neighbor_loo(X, y, k)
    best_accuracy = calculate_accuracy(y_true, y_pred)
    log = [(set(selected_features), best_accuracy)]

    print("Beginning search.")

    while len(selected_features) > 1:
        worst_feature = None
        for feature in selected_features:
            current_features = [f for f in selected_features if f != feature]
            X_subset = X[:, current_features]
            y_true, y_pred = nearest_neighbor_loo(X_subset, y, k)
            accuracy = calculate_accuracy(y_true, y_pred)
            print(f"Using feature(s) {set(current_features)} accuracy is {accuracy*100:.1f}%")
            log.append((set(current_features), accuracy))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                worst_feature = feature
        if worst_feature is not None:
            selected_features.remove(worst_feature)
        else:
            break

    return selected_features, best_accuracy, log

# plot accuracy
def plot_accuracies(log, title,file_path):
    feature_sets = [str(features) for features, acc in log]
    accuracies = [acc * 100 for features, acc in log]
    range_len = 10
    x_ticks = range(len(feature_sets))
    x_ticks_step = max(1, len(feature_sets) // range_len)
    x_ticks_labels = [feature_sets[i] for i in range(0, len(feature_sets), x_ticks_step)]
    x_ticks_positions = [i for i in range(0, len(feature_sets), x_ticks_step)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.bar(x_ticks, accuracies, color='b', alpha=0.6, label='Bar Plot')
    ax1.set_xlabel('Feature Sets')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_xticks(x_ticks_positions)
    ax1.set_xticklabels(x_ticks_labels, rotation=90)
    ax1.set_title(f'{title} - Bar Plot')

    ax2.plot(x_ticks, accuracies, color='r', marker='o', label='Line Plot')
    ax2.set_xlabel('Feature Sets')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xticks(x_ticks_positions)
    ax2.set_xticklabels(x_ticks_labels, rotation=90)
    ax2.set_title(f'{title} - Line Plot')

    fig.tight_layout()
    plt.suptitle(f'Accuracy of Different Feature Sets - {title}', y=1.05)
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))
    # plt.show()
    plt.savefig(f'{title}_{file_path}.pdf')
    print(f'Graph saved as {title}_{file_path}.pdf')
    plt.close()

# Function to save log data to CSV
def save_log_to_csv(log, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Feature Set', 'Accuracy'])
        for features, acc in log:
            writer.writerow([str(features), acc])

# Main function to load data and perform feature selection
def main():
    file_path = input('Type in the name of the file to test below:-')
    # loading the data from the dataset using the load_data function
    data = load_data(file_path)
    # seperatig the features from the labels in the dataset.
    X, y = separate_features_labels(data)
    n_instances, n_features = X.shape

    print("Welcome to Parth's Feature Selection Algorithm.")
    # print("Type in the name of the file to test :")
    print("Type the number of the algorithm you want to run.")
    print("\t 1.Forward Selection")
    print("\t 2.Backward Elimination")

    # creating a switch case like choice for the choice of the algorithm
    algorithm_choice = int(input())

    if algorithm_choice == 1:
        print(f"\nThis dataset has {n_features} features (not including the class attribute), with {n_instances} instances.")
        # Running the intial nearest_neighbor using the Leave-One-Out evaluation method
        y_true, y_pred = nearest_neighbor_loo(X, y)

        # calculating the accuracy of the intial predictions
        accuracy = calculate_accuracy(y_true, y_pred)

        print(f"Running nearest neighbor with all {n_features} features, using “leaving-one-out” evaluation, I get an accuracy of {accuracy*100:.1f}%")
        start_time = time.time()

        # run the feature search algorithm
        best_features, best_accuracy, log = forward_selection(X, y)

        end_time = time.time()
        run_time = start_time - end_time
        print(f"Excuetion time using forward selection for {file_path} is:{run_time}")

        # Save the output in the log file in a csv format
        save_log_to_csv(log, f'forward_selection_{file_path}_log.csv')
        # plot the accuracy to track the performance of the algorithm
        plot_accuracies(log, 'Forward Selection',file_path)

    elif algorithm_choice == 2:
        print(f"\nThis dataset has {n_features} features (not including the class attribute), with {n_instances} instances.")

        # Running the intial nearest_neighbor using the Leave-One-Out evaluation method
        y_true, y_pred = nearest_neighbor_loo(X, y)

        # calculating the accuracy of the intial predictions
        accuracy = calculate_accuracy(y_true, y_pred)

        print(f"Running nearest neighbor with all {n_features} features, using “leaving-one-out” evaluation, I get an accuracy of {accuracy*100:.1f}%")
        
        start_time = time.time()

        # calculating the accuracy of the intial predictions
        best_features, best_accuracy, log = backward_elimination(X, y)

        end_time = time.time()
        run_time = end_time - start_time
        print(f"Execution time using backward elimination for {file_path} is:{run_time}")

        # Save the output in the log file in a csv format
        save_log_to_csv(log, f'backward_elimination_{file_path}_log.csv')
        # plot the accuracy to track the performance of the algorithm
        plot_accuracies(log, 'Backward Elimination',file_path)

    else:
        print("Invalid choice.")
        return

    print(f"Finished search!! The best feature subset is {set(best_features)}, which has an accuracy of {best_accuracy*100:.1f}%")

if __name__ == "__main__":
    main()
