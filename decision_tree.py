
from collections import Counter
from math import log2


# -------------------- Student Performance Dataset --------------------

data = [
    {"StudyHours": "High",   "Attendance": "Good", "PreviousGrade": "High",   "Internet": "Yes", "Pass": "Yes"},
    {"StudyHours": "High",   "Attendance": "Good", "PreviousGrade": "Low",    "Internet": "Yes", "Pass": "Yes"},
    {"StudyHours": "Low",    "Attendance": "Poor", "PreviousGrade": "Low",    "Internet": "No",  "Pass": "No"},
    {"StudyHours": "Medium", "Attendance": "Good", "PreviousGrade": "Medium", "Internet": "Yes", "Pass": "Yes"},
    {"StudyHours": "Low",    "Attendance": "Good", "PreviousGrade": "Low",    "Internet": "Yes", "Pass": "No"},
    {"StudyHours": "High",   "Attendance": "Poor", "PreviousGrade": "High",   "Internet": "No",  "Pass": "Yes"},
    {"StudyHours": "Medium", "Attendance": "Poor", "PreviousGrade": "Medium", "Internet": "Yes", "Pass": "No"},
    {"StudyHours": "High",   "Attendance": "Good", "PreviousGrade": "Medium", "Internet": "No",  "Pass": "Yes"},
    {"StudyHours": "Low",    "Attendance": "Poor", "PreviousGrade": "High",   "Internet": "Yes", "Pass": "No"},
    {"StudyHours": "Medium", "Attendance": "Good", "PreviousGrade": "Low",    "Internet": "No",  "Pass": "No"},
    {"StudyHours": "High",   "Attendance": "Poor", "PreviousGrade": "Medium", "Internet": "Yes", "Pass": "Yes"},
    {"StudyHours": "Low",    "Attendance": "Good", "PreviousGrade": "Medium", "Internet": "No",  "Pass": "No"},
    {"StudyHours": "Medium", "Attendance": "Good", "PreviousGrade": "High",   "Internet": "Yes", "Pass": "Yes"},
    {"StudyHours": "High",   "Attendance": "Poor", "PreviousGrade": "Low",    "Internet": "No",  "Pass": "No"},
]

TARGET = "Pass"
FEATURES = [column for column in data[0] if column != TARGET]


# -------------------- ID3 calculations --------------------

def entropy(labels):
    
    counts = Counter(labels)
    total = len(labels)

    return -sum(
        (count / total) * log2(count / total)
        for count in counts.values()
    )


def information_gain(rows, feature, target):
    """Calculate information gain and details for a feature split."""
    parent_entropy = entropy([row[target] for row in rows])
    weighted_child_entropy = 0
    details = []

    values = sorted({row[feature] for row in rows})

    for value in values:
        subset = [row for row in rows if row[feature] == value]
        subset_entropy = entropy([row[target] for row in subset])
        weight = len(subset) / len(rows)
        weighted_child_entropy += weight * subset_entropy

        details.append((value, len(subset), subset_entropy, weight))

    gain = parent_entropy - weighted_child_entropy
    return parent_entropy, weighted_child_entropy, gain, details


def id3(rows, features, target, depth=0):
    """Build a decision tree and print calculations at each node."""
    labels = [row[target] for row in rows]

    # Stop if all examples have the same class.
    if len(set(labels)) == 1:
        return labels[0]

    # Stop if no features remain; use the majority class.
    if not features:
        return Counter(labels).most_common(1)[0][0]

    parent_entropy = entropy(labels)
    indent = "    " * depth

    print(f"\n{indent}Node: {len(rows)} records | Entropy = {parent_entropy:.4f}")
    print(f"{indent}{'Feature':<16}{'Weighted Entropy':>20}{'Information Gain':>20}")
    print(f"{indent}{'-' * 56}")

    gains = {}

    for feature in features:
        _, weighted_entropy, gain, details = information_gain(
            rows, feature, target
        )
        gains[feature] = gain

        print(f"{indent}{feature:<16}{weighted_entropy:>20.4f}{gain:>20.4f}")

        for value, count, subset_entropy, weight in details:
            print(
                f"{indent}  {value:<14} n={count:<2} "
                f"entropy={subset_entropy:.4f}, weight={weight:.4f}"
            )

    # Select the feature with the highest information gain.
    best_feature = max(gains, key=gains.get)
    print(f"{indent}Best split: {best_feature} (gain = {gains[best_feature]:.4f})")

    tree = {best_feature: {}}
    remaining_features = [f for f in features if f != best_feature]
    majority_class = Counter(labels).most_common(1)[0][0]

    for value in sorted({row[best_feature] for row in rows}):
        subset = [row for row in rows if row[best_feature] == value]

        if not subset:
            tree[best_feature][value] = majority_class
        else:
            tree[best_feature][value] = id3(
                subset, remaining_features, target, depth + 1
            )

    tree[best_feature]["_default"] = majority_class
    return tree


def predict(tree, example):
    """Predict the target label for one example."""
    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))
    branches = tree[feature]
    value = example.get(feature, "_missing")
    subtree = branches.get(value, branches["_default"])

    return predict(subtree, example)


def print_tree(tree, indent=""):
    """Display the tree in a readable format."""
    if not isinstance(tree, dict):
        print(f"{indent}→ {tree}")
        return

    feature = next(iter(tree))

    for value, subtree in tree[feature].items():
        if value == "_default":
            continue
        print(f"{indent}{feature} = {value}:")
        print_tree(subtree, indent + "    ")


# -------------------- Display data table first --------------------

columns = FEATURES + [TARGET]

print("=" * 82)
print("STUDENT PERFORMANCE DATASET")
print("=" * 82)
print(" | ".join(f"{column:<16}" for column in columns))
print("-" * 100)

for row in data:
    print(" | ".join(f"{row[column]:<16}" for column in columns))


# -------------------- Train and print solution --------------------

print("\n" + "=" * 82)
print("ENTROPY AND INFORMATION GAIN CALCULATIONS")
print("=" * 82)

tree = id3(data, FEATURES, TARGET)

print("\n" + "=" * 82)
print("FINAL DECISION TREE")
print("=" * 82)
print_tree(tree)

test_example = {
    "StudyHours": "High",
    "Attendance": "Good",
    "PreviousGrade": "Medium",
    "Internet": "Yes",
}

print("\nTest example:", test_example)
print("Predicted class:", predict(tree, test_example))
