import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from matplotlib.font_manager import FontProperties
import time
from skimage.transform import resize

def prepare_data_set():
    dataset_folder = "../dataset/"
    all_files = os.listdir(dataset_folder)
    samples = []
    for file in all_files:
        if file.endswith(".txt"):
            samples.append(file)
    data = []
    labels = []
    sample_number = 0
    original_samples = []
    augmented_samples = []
    original_labels = []
    augmented_labels = []
    for sample in samples:
        sample_path = dataset_folder + sample;
        raw_sample = read_file_content(sample_path);
        parsed_sample = json.loads(raw_sample);
        raw_points = parsed_sample["points"]
        transformed_sample = transform_emoji_points(raw_points)
        label = parsed_sample["emoji"]
        data.append(transformed_sample)
        labels.append(label)
        original_samples.append(transformed_sample)
        original_labels.append(label)
        # We improve our data set by generating more samples based on the original but with some modifications
        # using the original data on R2 in order to guarantee the output generated contains a similar number of points 
        # however, some augmentations are easier to implement if we use the transformed version, like horizontal flips
        transformed_augmentations = augment_transformed_sample(transformed_sample)
        for transformed_augmentation in transformed_augmentations:
            data.append(transformed_augmentation)
            labels.append(label)
            augmented_samples.append(transformed_augmentation)
            augmented_labels.append(label)
        raw_augmentations = augment_raw_sample(raw_points)
        for augmentation in raw_augmentations:
            transformed_raw_augmentation = transform_emoji_points(augmentation)
            data.append(transformed_raw_augmentation)
            labels.append(label)
            augmented_samples.append(transformed_raw_augmentation)
            augmented_labels.append(label)
        sample_number += 1
    print("    Samples and augmented samples ready, let's transform them into features")
    print("    Number of  original samples = " + str(len(original_samples)))
    print("    Number of augmented samples = " + str(len(augmented_samples)))
    print("    Total  number  of   samples = " + str(len(data)))
    data = crop_data_samples(data)
    return (data, labels, original_samples, original_labels, augmented_samples, augmented_labels)

def crop_data_samples(data_samples):
    cropped_samples = []
    for sample in data_samples:
        clean_sample = map_to_zero_or_one(sample)
        cropped_sample = crop_data_sample(clean_sample)
        cropped_and_resized_sample = resize(cropped_sample, (100, 100), anti_aliasing = False)
        clean_final_sample = map_to_zero_or_one(cropped_and_resized_sample)
        # We either ensure we can replicate map_to_zero and other matrix operations from TS or we will have to implement this 
        # using a different approach
        #print(f'Sample number of points = {np.count_nonzero(sample)}')
        #print(f'Clean sample  of points = {np.count_nonzero(clean_sample)}')
        #print(f'Croppedresize of points = {np.count_nonzero(cropped_and_resized_sample)}')
        #print(f'Final clean # of points = {np.count_nonzero(clean_final_sample)}')
        #show_sample(clean_final_sample, 20)
        cropped_samples.append(clean_final_sample)
    return cropped_samples

def crop_data_sample(sample):
    r = sample.any(1)
    if r.any():
        m,n = sample.shape
        c = sample.any(0)
        out = sample[r.argmax():m-r[::-1].argmax(), c.argmax():n-c[::-1].argmax()]
    else:
        out = np.empty((0,0),dtype=int)
    return  out

def normalized_value (x):
    if x >= 0.01:
        return 1
    else:
        return 0
map_to_zero_or_one =  np.vectorize(normalized_value, otypes = [float])

def augment_raw_sample(raw_sample):
    augmented_samples = []
    rotate_values = [-5, -10, -15, -20, -25, -30, -35, -40, -45, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    for rotation_angle in rotate_values:
        rotated_sample = [];
        for points in raw_sample:
            new_x, new_y = rotate_point_around_center(points["x"], points["y"], rotation_angle)
            rotated_sample.append({ "x": new_x, "y": new_y })
        augmented_samples.append(rotated_sample)
    return augmented_samples

def rotate_point_around_center(x, y, degrees):
    angle = np.radians(degrees)
    point_to_be_rotated = (x, y)    
    center_point = (200, 200)
    xnew = np.cos(angle)*(point_to_be_rotated[0] - center_point[0]) - np.sin(angle)*(point_to_be_rotated[1] - center_point[1]) + center_point[0]
    ynew = np.sin(angle)*(point_to_be_rotated[0] - center_point[0]) + np.cos(angle)*(point_to_be_rotated[1] - center_point[1]) + center_point[1]
    return (round(xnew,2),round(ynew,2))

def augment_transformed_sample(sample):
    augmented_samples = []
    sample_flipped_horizontally = np.fliplr(sample)
    augmented_samples.append(sample_flipped_horizontally)
    return augmented_samples

samples_shown = 0
def show_sample(sample, samples_to_show = 1):
    global samples_shown
    if samples_shown < samples_to_show:
        plt.imshow(sample, cmap='gray')
        plt.show()
        samples_shown += 1

def show_sample_if_contains_artifacts(sample):
    for x in sample:
        for y in x:
            if (y != 0.0 and y != 1.0):
                print("☢️☢️☢️  ARTIFACT DETECTED")
                show_sample(sample)
                return

def sample_is_empty(sample):
    return not sample_is_not_empty(sample)

def sample_is_not_empty(sample):
    return np.count_nonzero(sample) > 0

def read_file_content(path):
    file = open(path, "r")
    content = file.readline()
    file.close();
    return content;

def transform_emoji_points(points):
    input_size = 400
    black_and_white_points = np.zeros((input_size, input_size));
    for point in points:
        x = int(point["x"])
        y = int(point["y"])
        point_inside_the_matrix = x < input_size and y < input_size and x >= 0 and y >= 0
        if (point_inside_the_matrix):
            black_and_white_points[y][x] = 1;
    return black_and_white_points;

def train_model(data, labels):
    print("⏲  Starting the training process")
    flattened_data = []
    for sample in data:
        flattened_data.append(sample.flatten())
    data_train, data_test, labels_train, labels_test = train_test_split(flattened_data, labels, train_size=0.8, random_state=0)
    print("🖖 Dataset divided into: ")
    print("     Data  train size: ", len(data_train))
    print("     Label train size: ", len(labels_train))
    print("     Data   test size: ", len(data_test))
    print("     Label  test size: ", len(labels_test))
    logistic_regression = initialize_and_fit_logistic_regression_model(data_train, labels_train)
    test_element = data_test[0].reshape(1,-1)
    expected_label_for_test_element = labels_test[0]
    # Here we can find all the model classes logistic_regression.classes_
    # We can use logistic_regression.predict_log_proba(test_element) to get the probability value
    test_prediction = logistic_regression.predict(test_element)
    print(f'    Model tested after trainign expecting {expected_label_for_test_element} and got {test_prediction}')
    return (logistic_regression, data_train, data_test, labels_train, labels_test)

def initialize_and_fit_logistic_regression_model(data_train, labels_train):
    logistic_regression = LogisticRegression(n_jobs = os.cpu_count(), verbose = True)
    print("⌛️ Training the model")
    start_time = time.time()
    logistic_regression.fit(data_train, labels_train)
    end_time = time.time()
    print(f'✅ Model training completed. Training time {end_time - start_time} seconds')
    return logistic_regression

def evaluate_model_accuracy(experiment_name, model, data_train, data_test, labels_train, labels_test):
    print("📏 Evaluating model accuracy using the test and train data")
    print(f'    Train data size = {len(data_train)}. Train data labels = {len(labels_train)}')
    print(f'    Test data size = {len(data_test)}. Train data labels = {len(labels_test)}')
    train_score = model.score(data_train, labels_train)
    test_score = model.score(data_test, labels_test)
    test_predictions = model.predict(data_test)
    print(f'    Test  score = {test_score}')
    print(f'    Train score = {train_score}')
    generate_confusion_matrix(experiment_name, model, labels_test, test_score, test_predictions)
    generate_classification_text_report(experiment_name, labels_test, test_predictions)
    test_probablity_predictions = model.predict_proba(data_test)
    generate_probability_text_report(experiment_name, model, labels_test, test_probablity_predictions)

def generate_classification_text_report(experiment_name, labels_test, test_predictions):
    report_file = open(f'./metrics/test_prediction_report_{experiment_name}.txt', "w")
    test_data_set_size = len(test_predictions) 
    report_file.write(f'📊 Test prediction report for {test_data_set_size} elements\n')
    for index in range(test_data_set_size):
        result = "✅" if labels_test[index] == test_predictions[index] else "❌"
        individual_report = f'{result} => Expected: {labels_test[index]} - Got: {test_predictions[index]} \n'
        report_file.write(individual_report)
    report_file.close()

def generate_probability_text_report(experiment_name, model, labels_test, test_probability_predictions):
    print(f"📏 Evaluating model accuracy based on classification probability")
    all_labels = model.classes_
    report_file = open(f'./metrics/test_probability_prediction_report_{experiment_name}.txt', "w")
    test_data_set_size = len(test_probability_predictions) 
    report_file.write(f'📊 Test probability prediction report for {test_data_set_size} elements\n')
    predictions_above_90_percent = 0
    predictions_above_80_percent = 0
    predictions_above_70_percent = 0
    predictions_above_60_percent = 0
    predictions_above_50_percent = 0
    correct_predictions_above_90_percent = 0
    correct_predictions_above_80_percent = 0
    correct_predictions_above_70_percent = 0
    correct_predictions_above_60_percent = 0
    correct_predictions_above_50_percent = 0
    correct_perdictions_per_label = dict.fromkeys(all_labels, 0)
    predictions_per_label = dict.fromkeys(all_labels, 0)
    number_of_occurences_per_label = dict.fromkeys(all_labels, 0)
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
        expected_label = labels_test[index]
        # We save how many occurences per labels we have
        number_of_occurences_per_label[expected_label] +=1
        # We save how many times we predicted one label even if it's a failure
        predictions_per_label[best_prediction_label] += 1
        prediction_correct = expected_label == best_prediction_label
        if (prediction_correct):
            correct_perdictions_per_label[best_prediction_label] += 1
        if (best_prediction_result >= 0.9):
            predictions_above_90_percent += 1
            if (prediction_correct):
                correct_predictions_above_90_percent += 1
        if (best_prediction_result >= 0.8):
            predictions_above_80_percent += 1
            if (prediction_correct):
                correct_predictions_above_80_percent += 1
        if (best_prediction_result >= 0.7):
            predictions_above_70_percent += 1
            if (prediction_correct):
                correct_predictions_above_70_percent += 1
        if (best_prediction_result >= 0.6):
            predictions_above_60_percent += 1
            if (prediction_correct):
                correct_predictions_above_60_percent += 1
        if (best_prediction_result >= 0.5):
            predictions_above_50_percent += 1
            if (prediction_correct):
                correct_predictions_above_50_percent += 1
        header = "✅" if prediction_correct else "❌"
        individual_report = f'{header} => Expected: {labels_test[index]} - Got: {best_prediction_label}. Probability = {best_prediction_result} \n'
        report_file.write(individual_report)
    report_file.write("\n------------- Prediction probability -------------\n")
    report_file.write(f'Predictions above 90% = {predictions_above_90_percent} - {(predictions_above_90_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Correct pre above 90% = {correct_predictions_above_90_percent} - {(correct_predictions_above_90_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Predictions above 80% = {predictions_above_80_percent} - {(predictions_above_80_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Correct pre above 80% = {correct_predictions_above_80_percent} - {(correct_predictions_above_80_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Predictions above 70% = {predictions_above_70_percent} - {(predictions_above_70_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Correct pre above 70% = {correct_predictions_above_70_percent} - {(correct_predictions_above_70_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Predictions above 60% = {predictions_above_60_percent} - {(predictions_above_60_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Correct pre above 60% = {correct_predictions_above_60_percent} - {(correct_predictions_above_60_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Predictions above 50% = {predictions_above_50_percent} - {(predictions_above_50_percent / test_data_set_size) * 100}%\n')
    report_file.write(f'Correct pre above 50% = {correct_predictions_above_50_percent} - {(correct_predictions_above_50_percent / test_data_set_size) * 100}%\n')
    report_file.write("\n------------- Prediction per label -------------\n")
    for label in all_labels:
        if (number_of_occurences_per_label[label] == 0):
            print(f'☢️☢️☢️  Label {label} has 0 predictions ☢️☢️☢️')
        else:
            report_file.write(f'Correct predictions for {label} = {(correct_perdictions_per_label[label] / number_of_occurences_per_label[label]) * 100}%. Correct = {correct_perdictions_per_label[label]}. Total = {number_of_occurences_per_label[label]}\n')
    correct_prediction_percentage = dict.fromkeys(all_labels, "")
    for label in all_labels:
        if (number_of_occurences_per_label[label] != 0):
            correct_prediction_percentage[label] = f'{correct_perdictions_per_label[label] / number_of_occurences_per_label[label] * 100}%'
    print(f'    Success predictions per label:{correct_perdictions_per_label}' )
    print(f'    Total   predictions per label:{number_of_occurences_per_label}' )
    print(f'    Acc % per label :{correct_prediction_percentage}' )
    report_file.close()

def generate_confusion_matrix(experiment_name, model, labels_test, test_score, test_predictions):
    confusion_matrix = metrics.confusion_matrix(labels_test, test_predictions)
    plt.figure(figsize=(9,9))
    sns.heatmap(confusion_matrix, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    all_sample_title = 'Accuracy Score: {0}'.format(test_score)
    plt.title(all_sample_title, size = 15);
    plt.savefig(f'./metrics/confusion_matrix_{experiment_name}.png')
    print(f'    Confusion matrix index-emoji legend:')
    index = 0
    for label in model.classes_:
        print(f'        {index} - {label}')
        index += 1

def show_some_data_examples(data, labels, number_of_samples):
    print("🔍 Showing some data examples")
    for index, (image, label) in enumerate(zip(data[0:number_of_samples], labels[0:number_of_samples])):
        print(f'    Preparing visual representation of {label} for sample number: {index}')
        plt.imshow(image, cmap='gray')
        plt.show()

def main():
    print("😃 Initializing HWEmoji training script")
    print("🤓 Preparing trainig data using the files from /dataset")
    data, labels, original_samples, original_labels, augmented_samlpes, augmented_labels = prepare_data_set()
    print(f'📖 Data set ready with {len(data)} samples asociated to {len(labels)} labels')
    #show_some_data_examples(data, labels, 20)
    train_and_evaluate_accuracy_with_all_the_data(data, labels)
    train_and_evaluate_accuracy_with_augmented_samples_only(original_samples, original_labels, augmented_samlpes, augmented_labels)
    print("✅ Training completed")

def train_and_evaluate_accuracy_with_all_the_data(data, labels):
    model, data_train, data_test, labels_train, labels_test = train_model(data, labels)
    print(f'💪 Model trained with {len(data_train)} samples. Evaluating model accuracy')
    evaluate_model_accuracy("full_data_set", model, data_train, data_test, labels_train, labels_test)

def train_and_evaluate_accuracy_with_augmented_samples_only(original_samples, original_labels, augmented_samples, augmented_labels):
    print("⏲  Starting the training process with augmented samples only")
    flattened_augmented_data = []
    for augmented_sample in augmented_samples:
        flattened_augmented_data.append(augmented_sample.flatten())
    flattened_original_data = []
    for original_sample in original_samples:
        flattened_original_data.append(original_sample.flatten())
    model = initialize_and_fit_logistic_regression_model(flattened_augmented_data, augmented_labels)
    print(f'💪 Model trained with {len(augmented_samples)} augmented samples. Evaluating model accuracy')
    evaluate_model_accuracy("augmented_samples_training", model, flattened_augmented_data, flattened_original_data, augmented_labels, original_labels)

if __name__ == "__main__":
    main()
