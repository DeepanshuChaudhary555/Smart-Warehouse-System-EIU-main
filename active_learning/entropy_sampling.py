import numpy as np


def calculate_entropy(probabilities):

    probabilities = np.array(probabilities, dtype=np.float32)

    probabilities += 1e-10

    entropy = -np.sum(probabilities * np.log(probabilities))

    return float(entropy)


def entropy_filter(predictions, threshold=0.5):

    uncertain_samples = []

    for pred in predictions:

        entropy = calculate_entropy(pred["probs"])

        if entropy >= threshold:

            pred["entropy"] = entropy

            uncertain_samples.append(pred)

    return uncertain_samples