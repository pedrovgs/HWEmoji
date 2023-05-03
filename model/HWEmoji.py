import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics

def prepare_data_set():
    dataset_folder = "../dataset/"
    samples = os.listdir(dataset_folder)
    data = []
    labels = []
    for sample in samples:
        sample_path = dataset_folder + sample;
        raw_sample = read_file_content(sample_path);
        parsed_sample = json.loads(raw_sample);
        data.append(transform_emoji_points(parsed_sample["points"]))
        labels.append(parsed_sample["emoji"])
    return (data, labels)

def read_file_content(path):
    file = open(path, "r")
    content = file.readline()
    file.close();
    return content;

def transform_emoji_points(points):
    black_and_white_points = np.zeros((400, 400));
    for point in points:
        x = int(point["x"])
        y = int(point["y"])
        black_and_white_points[x][y] = 1;
    return black_and_white_points.flatten()

def train_model(data, labels):
    print("⏲  Starting the training process")
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.25, random_state=0)
    print("🖖 Dataset divided into: ")
    print("     Data  train size: ", len(data_train))
    print("     Label train size: ", len(labels_train))
    print("     Data   test size: ", len(data_test))
    print("     Label  test size: ", len(labels_test))
    logistic_regression = LogisticRegression(solver = 'lbfgs')
    print("⌛️ Training the model")
    logistic_regression.fit(data_train, labels_train)
    print("✅ Model training completed. Evaluating for one element")
    test_element = data_test[0].reshape(1,-1)
    expected_label_for_test_element = labels_test[0]
    # Here we can find all the model classes logistic_regression.classes_
    # We can use logistic_regression.predict_log_proba(test_element) to get the probability value
    test_prediction = logistic_regression.predict(test_element)
    print(f'    Model tested after trainign expecting {expected_label_for_test_element} and got ${test_prediction}')
    return (logistic_regression, data_train, data_test, labels_train, labels_test)

def evaluate_model_accuracy(model, data_train, data_test, labels_train, labels_test):
    print("📏 Evaluating model accuracy using the test and train data")
    train_score = model.score(data_train, labels_train)
    test_score = model.score(data_test, labels_test)
    print(f'    Test  score = {test_score}')
    print(f'    Train score = {train_score}')

def main():
    print("😃 Initializing HWEmoji training script")
    print("🤓 Preparing trainig data using the files from /dataset")
    data, labels = prepare_data_set()
    model, data_train, data_test, labels_train, labels_test = train_model(data, labels)
    print(f'💪 Model trained with {len(data_train)} samples. Evaluating model accuracy')
    evaluate_model_accuracy(model, data_train, data_test, labels_train, labels_test)
    print("✅ Model updated and saved")



if __name__ == "__main__":
    main()
