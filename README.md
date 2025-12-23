import pandas as pd
import math
from pprint import pprint # For nicely printing the tree

# 1. Simpler: Data is now inside the script
data = pd.read_csv('buy_computer.csv')

def entropy(df, target_attr):
    """Calculates the entropy of a dataset."""
    values = df[target_attr].unique()
    entropy_val = 0
    total = len(df)
    
    for val in values:
        p = len(df[df[target_attr] == val]) / total
        if p > 0:
            entropy_val -= p * math.log2(p)
            
    # ***CORRECTION:*** 'return' must be OUTSIDE the loop
    return entropy_val

def info_gain(df, attr, target_attr):
    """Calculates the information gain of an attribute."""
    total_entropy = entropy(df, target_attr)
    
    values = df[attr].unique()
    subset_entropy = 0
    total = len(df)
    
    for val in values:
        subset = df[df[attr] == val]
        weight = len(subset) / total
        subset_entropy += weight * entropy(subset, target_attr)
        
    # ***CORRECTION:*** 'gain' calculation and 'return' must be OUTSIDE the loop
    gain = total_entropy - subset_entropy
    return gain

def id3(df, target_attr, attributes):
    """Builds the decision tree recursively."""
    
    # Base Case 1: If all target values are the same (pure node)
    if len(df[target_attr].unique()) == 1:
        return df[target_attr].iloc[0]
    
    # Base Case 2: If there are no more attributes to split on
    if len(attributes) == 0:
        return df[target_attr].mode()[0]  # Return majority vote
    
    # --- Recursive Step ---
    
    # Find the best attribute to split on
    gains = {attr: info_gain(df, attr, target_attr) for attr in attributes}
    best_attr = max(gains, key=gains.get)
    
    # Create the root node for this tree
    tree = {best_attr: {}}
    
    # Remove this attribute from the list for the next recursion
    new_attrs = [a for a in attributes if a != best_attr]
    
    # Create branches for each value of the best attribute
    for val in df[best_attr].unique():
        subset = df[df[best_attr] == val]
        
        # Recursively call id3 on the subset
        tree[best_attr][val] = id3(subset, target_attr, new_attrs)
        
    # ***CORRECTION:*** 'return' must be OUTSIDE the loop
    return tree

def predict(tree, sample):
    """Predicts the class for a new sample."""
    # If we are at a leaf node (a final answer)
    if not isinstance(tree, dict):
        return tree
    
    # Get the attribute for the current node
    attribute = list(tree.keys())[0]
    subtree = tree[attribute]
    
    # Get the sample's value for that attribute
    value = sample.get(attribute)
    
    # Check if this branch exists
    if value in subtree:
        # Recurse down the branch
        return predict(subtree[value], sample)
    else:
        # This value was not seen in training
        return "Unknown" # Or handle as a different case

# --- Main Program ---
target_attribute = 'buys_computer'
all_attributes = list(data.columns)
all_attributes.remove(target_attribute)

# Build the tree
tree = id3(data, target_attribute, all_attributes)

print("Generated Decision Tree:")
pprint(tree) # Use pprint for a much cleaner output

# Test with a new sample
new_sample = {
    'age': 'youth',
    'income': 'medium',
    'student': 'yes',
    'credit_rating': 'fair'
}

predicted_class = predict(tree, new_sample)
print("\nNew Sample:", new_sample)
print("Predicted Class:", predicted_class)
