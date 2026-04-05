#import codecademylib3_seaborn
# Step 1: Import the breast cancer dataset loader
from sklearn.datasets import load_breast_cancer

# Load the data
breast_cancer_data = load_breast_cancer()

# Step 2: Explore the data
print("First data point features:")
print(breast_cancer_data.data[0])

print("\nFeature names:")
print(breast_cancer_data.feature_names)

print("\nAll data points:")
print(breast_cancer_data.data)

# Step 3: Check the target labels and names
print("\nTarget labels:")
print(breast_cancer_data.target)

print("\nTarget names:")
print(breast_cancer_data.target_names)

print("\nLabel of first data point:", breast_cancer_data.target[0])
print("Which corresponds to:", breast_cancer_data.target_names[breast_cancer_data.target[0]])

# Step 4: Import train_test_split
from sklearn.model_selection import train_test_split

# Step 5 and 6: Split the data into training and validation sets
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data,
    breast_cancer_data.target,
    test_size=0.2,
    random_state=100
)

# Step 7: Confirm the split
print("\nLength of training data:", len(training_data))
print("Length of training labels:", len(training_labels))

# Step 8: Import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

# Step 9: Create classifier with n_neighbors=3
classifier = KNeighborsClassifier(n_neighbors=3)

# Step 10: Train the classifier
classifier.fit(training_data, training_labels)

# Step 11: Check accuracy on validation set
accuracy = classifier.score(validation_data, validation_labels)
print("\nValidation accuracy with k=3:", accuracy)

# Step 12: Test k from 1 to 100 and print accuracies
print("\nValidation accuracies for k from 1 to 100:")
for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracy = classifier.score(validation_data, validation_labels)
    print(f"k={k}: accuracy={accuracy:.4f}")

# Step 13: Import matplotlib for plotting
import matplotlib.pyplot as plt

# Step 14: Create list of k values
k_list = list(range(1, 101))

# Step 15: Collect accuracies in a list
accuracies = []
for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

# Step 16: Plot the results
plt.plot(k_list, accuracies)
plt.show()

# Step 17: Add labels and title
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()

# Step 18: (Optional) See how different random_state values affect results
print("\n" + "="*60)
print("TESTANDO DIFERENTES RANDOM_STATE VALUES")
print("="*60)

random_states = [1, 42, 100, 123, 999]
results = {}

for random_state in random_states:
    # Split with different random_state
    training_data_rs, validation_data_rs, training_labels_rs, validation_labels_rs = train_test_split(
        breast_cancer_data.data,
        breast_cancer_data.target,
        test_size=0.2,
        random_state=random_state
    )
    
    # Train with k=3
    classifier_rs = KNeighborsClassifier(n_neighbors=3)
    classifier_rs.fit(training_data_rs, training_labels_rs)
    accuracy_rs = classifier_rs.score(validation_data_rs, validation_labels_rs)
    
    results[random_state] = accuracy_rs
    print(f"random_state={random_state:3d} → accuracy={accuracy_rs:.4f}")

print("\nMédia das acurácias:", f"{sum(results.values()) / len(results):.4f}")
print("\nO random_state controla COMO os dados são divididos entre treino/validação.")
print("Diferentes divisões podem gerar diferentes resultados!")
