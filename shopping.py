import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as file:
        contents_page = csv.reader(file)

        # Categorizing indices
        contents = []
        for content in contents_page:
            contents.append(content)

        first_row = contents[0]
        int_type = []
        float_type = []
        month_type = []
        visitor_type = []
        weekend_type = []
        label_type = []
        month = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4,
                 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9,
                 'Nov': 10, 'Dec': 11}

        for num, heading in enumerate(first_row):
            if heading in ['Administrative_Duration',
                           'Informational_Duration',
                           'ProductRelated_Duration',
                           'BounceRates', 'ExitRates',
                           'PageValues', 'SpecialDay']:
                float_type.append(num)
            elif heading in ['Administrative', 'Informational',
                             'ProductRelated',
                             'OperatingSystems',
                             'Browser', 'Region', 'TrafficType']:
                int_type.append(num)
            elif heading == 'Month':
                month_type.append(num)
            elif heading == 'Weekend':
                weekend_type.append(num)
            elif heading == 'VisitorType':
                visitor_type.append(num)
            elif heading == 'Revenue':
                label_type.append(num)
        contents.remove(first_row)

        # Finding evidences and labels
        evidences = []
        labels = []
        for line in contents:
            row = []
            for num, value in enumerate(line):
                if num in int_type:
                    row.append(int(value))
                elif num in float_type:
                    row.append(float(value))
                elif num in month_type:
                    row.append(month[value])
                elif num in visitor_type:
                    row.append(int(1)) if value == 'Returning_Visitor' else \
                        row.append(int(0))
                elif num in weekend_type:
                    row.append(int(1)) if value == 'TRUE' else \
                        row.append(int(0))
                elif num in label_type:
                    labels.append(int(1)) if value == 'TRUE' else \
                        labels.append(int(0))
            evidences.append(row)
    result = (evidences, labels)
    return result


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    positive_labels = 0
    negative_labels = 0
    for num, item in enumerate(labels):
        if item == 1 and predictions[num] == 1:
            positive_labels += 1
        if item == 0 and predictions[num] == 0:
            negative_labels += 1

    sensitivity = positive_labels / (list(labels).count(1))
    specificity = negative_labels / (list(labels).count(0))
    result = (sensitivity, specificity)
    return result


if __name__ == "__main__":
    main()
