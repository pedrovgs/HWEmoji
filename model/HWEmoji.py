import os
import json
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy.ndimage import rotate
import time
import skimage
from skimage.transform import AffineTransform

def prepare_data_set():
    dataset_folder = "../dataset/"
    samples = os.listdir(dataset_folder)
    data = []
    labels = []
    for sample in samples:
        sample_path = dataset_folder + sample;
        raw_sample = read_file_content(sample_path);
        parsed_sample = json.loads(raw_sample);
        transformed_sample = transform_emoji_points(parsed_sample["points"])
        label = parsed_sample["emoji"]
        data.append(transformed_sample)
        labels.append(label)
        # We improve our data set by generating more samples based on the original but with some modifications
        augmentations = augment_sample(transformed_sample)
        for augmentation in augmentations:
            data.append(augmentation)
            labels.append(label)
        data = crop_data_samples(data)
    return (data, labels)

def crop_data_samples(data_samples):
    cropped_samples = []
    for sample in data_samples:
        cropped_sample = sample
        cropped_samples.append(cropped_sample)
    return cropped_samples

def augment_sample(sample):
    augmented_samples = []
    sample_flipped_horizontally = np.fliplr(sample)
    augmented_samples.append(sample_flipped_horizontally)
    sample_size = 400
    shift_values = [0.1, 0.08, 0.06, 0.04, 0.02]
    for shift in shift_values:
        right_shift_sample = np.roll(sample, int(sample_size * shift))
        augmented_samples.append(right_shift_sample)
        left_shift_sample = np.roll(sample, int(sample_size * shift * -1))
        augmented_samples.append(left_shift_sample)
        up_shift_sample = np.roll(sample, int(sample_size * shift * -1), axis=0)
        augmented_samples.append(up_shift_sample)
        down_shift_sample = np.roll(sample, int(sample_size * shift), axis=0)
        augmented_samples.append(down_shift_sample)
        down_and_right_shift_sample = np.roll(down_shift_sample, int(sample_size * shift))
        augmented_samples.append(down_and_right_shift_sample)
        down_and_left_shift_sample = np.roll(down_shift_sample, int(sample_size * shift * -1))
        augmented_samples.append(down_and_left_shift_sample)
        up_and_right_shift_sample = np.roll(up_shift_sample, int(sample_size * shift))
        augmented_samples.append(up_and_right_shift_sample)
        up_and_left_shift_sample = np.roll(up_shift_sample, int(sample_size * shift * -1))
        augmented_samples.append(up_and_left_shift_sample)
    rotate_values = [5, 10, 15, 20, 25, 30, 35, 40, 45]
    for rotation_angle in rotate_values:
        rotated_sample = rotate(sample, angle=rotation_angle, reshape=False)
        augmented_samples.append(rotated_sample)
    scale_values = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]
    for scale in scale_values:
        scale_transform = AffineTransform(scale = scale)
        rescaled_sample = skimage.transform.warp(sample, scale_transform.inverse)
        augmented_samples.append(rescaled_sample)
    return augmented_samples

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
        black_and_white_points[y][x] = 1;
    return black_and_white_points;

def train_model(data, labels):
    print("⏲  Starting the training process")
    flattened_data = []
    for sample in data:
        flattened_data.append(sample.flatten())
    data_train, data_test, labels_train, labels_test = train_test_split(flattened_data, labels, train_size=0.9, random_state=0)
    print("🖖 Dataset divided into: ")
    print("     Data  train size: ", len(data_train))
    print("     Label train size: ", len(labels_train))
    print("     Data   test size: ", len(data_test))
    print("     Label  test size: ", len(labels_test))
    # solver = "saga", max_iter = 50 got the best results in terms of accuracy,
    # but training the model takes forever.
    logistic_regression = LogisticRegression(n_jobs = os.cpu_count(), verbose = True)
    print("⌛️ Training the model")
    start_time = time.time()
    logistic_regression.fit(data_train, labels_train)
    end_time = time.time()
    print(f'✅ Model training completed. Training time {end_time - start_time} seconds')
    test_element = data_test[0].reshape(1,-1)
    expected_label_for_test_element = labels_test[0]
    # Here we can find all the model classes logistic_regression.classes_
    # We can use logistic_regression.predict_log_proba(test_element) to get the probability value
    test_prediction = logistic_regression.predict(test_element)
    print(f'    Model tested after trainign expecting {expected_label_for_test_element} and got {test_prediction}')
    return (logistic_regression, data_train, data_test, labels_train, labels_test)

def evaluate_model_accuracy(model, data_train, data_test, labels_train, labels_test):
    print("📏 Evaluating model accuracy using the test and train data")
    train_score = model.score(data_train, labels_train)
    test_score = model.score(data_test, labels_test)
    test_predictions = model.predict(data_test)
    print(f'    Test  score = {test_score}')
    print(f'    Train score = {train_score}')
    generate_confusion_matrix(model, labels_test, test_score, test_predictions)
    generate_classification_text_report(labels_test, test_predictions)
    test_probablity_predictions = model.predict_proba(data_test)
    generate_probability_text_report(model, labels_test, test_probablity_predictions)

def generate_classification_text_report(labels_test, test_predictions):
    report_file = open("./metrics/test_prediction_report.txt", "w")
    test_data_set_size = len(test_predictions) 
    report_file.write(f'📊 Test prediction report for {test_data_set_size} elements\n')
    for index in range(test_data_set_size):
        result = "✅" if labels_test[index] == test_predictions[index] else "❌"
        individual_report = f'{result} => Expected: {labels_test[index]} - Got: {test_predictions[index]} \n'
        report_file.write(individual_report)
    report_file.close()

def generate_probability_text_report(model, labels_test, test_probability_predictions):
    print("📏 Evaluating model accuracy based on classification probability")
    all_labels = model.classes_
    report_file = open("./metrics/test_probability_prediction_report.txt", "w")
    test_data_set_size = len(test_probability_predictions) 
    report_file.write(f'📊 Test probability prediction report for {test_data_set_size} elements\n')
    predictions_above_90_percent = 0
    predictions_above_80_percent = 0
    predictions_above_70_percent = 0
    predictions_above_60_percent = 0
    predictions_above_50_percent = 0
    correct_perdictions_per_label = dict.fromkeys(all_labels, 0)
    predictions_per_label = dict.fromkeys(all_labels, 0)
    for index in range(test_data_set_size):
        prediction_result = test_probability_predictions[index]
        best_prediction_result = -1
        best_prediction_label = 0
        label_index = 0
        for label_index in range(len(all_labels)):
            label = model.classes_[label_index]
            label_probability = prediction_result[label_index]
            if label_probability > best_prediction_result:
                best_prediction_result = prediction_result[label_index]
                best_prediction_label = label
        predictions_per_label[best_prediction_label] += 1
        prediction_correct = labels_test[index] == best_prediction_label
        if (prediction_correct):
            correct_perdictions_per_label[best_prediction_label] += 1
        if (best_prediction_result >= 0.9):
            predictions_above_90_percent += 1
        if (best_prediction_result >= 0.8):
            predictions_above_80_percent += 1
        if (best_prediction_result >= 0.7):
            predictions_above_70_percent += 1
        if (best_prediction_result >= 0.6):
            predictions_above_60_percent += 1
        if (best_prediction_result >= 0.5):
            predictions_above_50_percent += 1
        header = "✅" if prediction_correct else "❌"
        individual_report = f'{header} => Expected: {labels_test[index]} - Got: {best_prediction_label}. Probability = {best_prediction_result} \n'
        report_file.write(individual_report)
    report_file.write("\n------------- Prediction probability -------------\n")
    report_file.write(f'Predictions above 90% = {predictions_above_90_percent} - {(predictions_above_90_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Predictions above 80% = {predictions_above_80_percent} - {(predictions_above_80_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Predictions above 70% = {predictions_above_70_percent} - {(predictions_above_70_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Predictions above 60% = {predictions_above_60_percent} - {(predictions_above_60_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Predictions above 50% = {predictions_above_50_percent} - {(predictions_above_50_percent / test_data_set_size) * 100}%\n')
    report_file.write("\n------------- Prediction per label -------------\n")
    for label in all_labels:
        report_file.write(f'Correct predictions for {label} = {(correct_perdictions_per_label[label] / predictions_per_label[label]) * 100}%. Correct = {correct_perdictions_per_label[label]}. Total = {predictions_per_label[label]}\n')
    correct_preciction_percentage = dict.fromkeys(all_labels, "")
    for label in all_labels:
        correct_preciction_percentage[label] = f'{correct_perdictions_per_label[label] / predictions_per_label[label] * 100}%'
    print(f'    Acc per label:{correct_perdictions_per_label}' )
    print(f'    Predictions per label:{predictions_per_label}' )
    print(f'    Acc % per label:{correct_preciction_percentage}' )
    report_file.close()

def generate_confusion_matrix(model, labels_test, test_score, test_predictions):
    confusion_matrix = metrics.confusion_matrix(labels_test, test_predictions)
    plt.figure(figsize=(9,9))
    sns.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(test_score)
    plt.title(all_sample_title, size = 15);
    plt.savefig("./metrics/confusion_matrix.png")
    print(f'    Confusion matrix index-emoji legend:')
    index = 0
    for label in model.classes_:
        print(f'        {index} - {label}')
        index += 1

def show_some_data_examples(data, labels, number_of_samples):
    print("🔍 Showing some data examples")
    for index, (image, label) in enumerate(zip(data[0:number_of_samples], labels[0:number_of_samples])):
        print(f'    Preparing visual representation of {label} for sample number: {index}')
        reshaped_image = np.reshape(image, (400,400))
        plt.imshow(reshaped_image)
        plt.show()

def main():
    print("😃 Initializing HWEmoji training script")
    print("🤓 Preparing trainig data using the files from /dataset")
    data, labels = prepare_data_set()
    #show_some_data_examples(data, labels, 10)
    model, data_train, data_test, labels_train, labels_test = train_model(data, labels)
    print(f'💪 Model trained with {len(data_train)} samples. Evaluating model accuracy')
    evaluate_model_accuracy(model, data_train, data_test, labels_train, labels_test)
    print("✅ Model updated and saved")



if __name__ == "__main__":
    main()
