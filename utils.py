import math
from random import shuffle

'''
Utility methods for machine learning
'''


# split data in to training, test and validation sets with given test and validation ratio
def split_class_data(rows, test_ratio, validation_ratio):
    total_length = len(rows)
    test_length = math.ceil(total_length * test_ratio)
    validation_length = math.ceil(total_length * validation_ratio)
    train_length = total_length - test_length - validation_length

    if test_length <= 0 or validation_length <= 0 or train_length <= 0:
        raise Exception("There is not enough data for training!")

    shuffle(rows)

    train_data = rows[0:train_length]
    test_data = rows[train_length:train_length + test_length]
    validation_data = rows[train_length + test_length:]

    return train_data, test_data, validation_data


# split data into training, test and validation sets
# stratify using given index if stratify is set to True
def split_data(rows, data_index, stratify=False, train=0.7, test_ratio=0.2, validation_ratio=0.1):
    if train <= 0 or test_ratio <= 0 or validation_ratio <= 0 or train + test_ratio + validation_ratio > 1.01:
        raise Exception("Please provide valid ratios for splitting data!")

    if stratify:
        distinct_classes = set(row[data_index] for row in rows)
        train_data = []
        test_data = []
        validation_data = []

        for c in distinct_classes:
            class_data = [row for row in rows if row[0] == c]
            class_train, class_test, class_validation = split_class_data(class_data, test_ratio, validation_ratio)
            train_data.extend(class_train)
            test_data.extend(class_test)
            validation_data.extend(class_validation)

        shuffle(train_data)
        shuffle(test_data)
        shuffle(validation_data)
    else:
        train_data, test_data, validation_data = split_class_data(rows, test_ratio, validation_ratio)

    return train_data, test_data, validation_data
