import os
import matplotlib.pyplot as plt
import pandas as pd

# Function to generate and save plot from a CSV file
def generate_plot(file_name):
    df = pd.read_csv(file_name, encoding='utf-8-sig')

    # Extract the number of features and accuracies for plotting
    features = df['Feature Set']
    accuracies = df['Accuracy']

    # Set range length for x-ticks
    range_len = 10
    x_ticks = range(len(features))
    x_ticks_step = max(1, len(features) // range_len)
    x_ticks_labels = [features[i] for i in range(0, len(features), x_ticks_step)]
    x_ticks_positions = [i for i in range(0, len(features), x_ticks_step)]

    plt.figure(figsize=(10, 5))
    plt.scatter(x_ticks, accuracies, color='blue', marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Model Performance (Accuracy)')
    plt.title(f'Model Performance vs. Number of Features\n{file_name}')
    plt.xticks(x_ticks_positions, x_ticks_labels, rotation=90)
    plt.grid(True)

    # Save the plot as a PDF
    output_file = f'{file_name}_scatter.pdf'
    plt.savefig(output_file)
    plt.close()

# Iterate over all CSV files in the current directory
for file_name in os.listdir('.'):
    if file_name.endswith('.csv'):
        generate_plot(file_name)
