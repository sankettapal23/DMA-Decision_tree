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


# -------------------- Modified tree (C4.5 style) calculations --------------------

def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    return abs(-sum((c / total) * log2(c / total) for c in counts.values()))


def split_metrics(rows, feature, target):
    """Return gain, split information, gain ratio and per-value details."""
    parent_entropy = entropy([row[target] for row in rows])
    weighted_child_entropy = 0
    split_info = 0
    details = []

    for value in sorted({row[feature] for row in rows}):
        subset = [row for row in rows if row[feature] == value]
        subset_entropy = entropy([row[target] for row in subset])
        weight = len(subset) / len(rows)
        weighted_child_entropy += weight * subset_entropy
        split_info -= weight * log2(weight)
        details.append((value, len(subset), subset_entropy, weight))

    gain = parent_entropy - weighted_child_entropy
    gain_ratio = gain / split_info if split_info > 0 else 0
    return gain, split_info, gain_ratio, details


def modified_id3(rows, features, target, depth=0):
    """Build a tree using gain ratio, restricted to above-average-gain features."""
    labels = [row[target] for row in rows]

    if len(set(labels)) == 1:
        return labels[0]
    if not features:
        return Counter(labels).most_common(1)[0][0]

    indent = "    " * depth
    print(f"\n{indent}Node: {len(rows)} records | Entropy = {entropy(labels):.4f}")
    print(f"{indent}{'Feature':<16}{'Gain':>10}{'SplitInfo':>12}{'GainRatio':>12}")
    print(f"{indent}{'-' * 50}")

    metrics = {}
    for feature in features:
        gain, split_info, gain_ratio, details = split_metrics(rows, feature, target)
        metrics[feature] = (gain, split_info, gain_ratio)
        print(f"{indent}{feature:<16}{gain:>10.4f}{split_info:>12.4f}{gain_ratio:>12.4f}")
        for value, count, subset_entropy, weight in details:
            print(f"{indent}  {value:<14} n={count:<2} "
                  f"entropy={subset_entropy:.4f}, weight={weight:.4f}")

    # C4.5 heuristic: only consider features whose gain is at least the average gain,
    # then pick the one with the highest gain ratio.
    avg_gain = sum(m[0] for m in metrics.values()) / len(metrics)
    candidates = [f for f in features if metrics[f][0] >= avg_gain - 1e-12]
    best_feature = max(candidates, key=lambda f: metrics[f][2])

    print(f"{indent}Average gain = {avg_gain:.4f} | Candidates: {', '.join(candidates)}")
    print(f"{indent}Best split: {best_feature} (gain ratio = {metrics[best_feature][2]:.4f})")

    tree = {best_feature: {}}
    remaining = [f for f in features if f != best_feature]
    majority_class = Counter(labels).most_common(1)[0][0]

    for value in sorted({row[best_feature] for row in rows}):
        subset = [row for row in rows if row[best_feature] == value]
        tree[best_feature][value] = modified_id3(subset, remaining, target, depth + 1)

    tree[best_feature]["_default"] = majority_class
    return tree


def predict(tree, example):
    if not isinstance(tree, dict):
        return tree
    feature = next(iter(tree))
    branches = tree[feature]
    subtree = branches.get(example.get(feature, "_missing"), branches["_default"])
    return predict(subtree, example)


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(f"{indent}→ {tree}")
        return
    feature = next(iter(tree))
    for value, subtree in tree[feature].items():
        if value == "_default":
            continue
        print(f"{indent}{feature} = {value}:")
        print_tree(subtree, indent + "    ")


def accuracy(tree, rows, target):
    correct = sum(predict(tree, row) == row[target] for row in rows)
    return correct / len(rows)


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
print("GAIN RATIO CALCULATIONS (MODIFIED DECISION TREE)")
print("=" * 82)

tree = modified_id3(data, FEATURES, TARGET)

print("\n" + "=" * 82)
print("FINAL MODIFIED DECISION TREE")
print("=" * 82)
print_tree(tree)

print(f"\nTraining accuracy: {accuracy(tree, data, TARGET):.2%}")

test_example = {
    "StudyHours": "High",
    "Attendance": "Good",
    "PreviousGrade": "Medium",
    "Internet": "Yes",
}
print("\nTest example:", test_example)
print("Predicted class:", predict(tree, test_example))
