import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import imageio.v2 as img
from sklearn.neighbors import KNeighborsClassifier


def get_dataset(csv_name):
    v_images, v_labels = [], []

    try:
        if csv_name not in ['val.csv', 'train.csv', 'test.csv']:
            raise ValueError('Incorrect file name!')
    except ValueError as e:
        print(str(e))

    folder_name = csv_name.split('.')[0] + '_images/'

    with open(csv_name, 'r') as read_file:
        csv_content = pd.read_csv(read_file)

        for idx, row in csv_content.iterrows():
            image = img.imread(folder_name + row['Image'])
            image = np.asarray(image)

            mean_img = image.mean(axis=(0, 1), dtype='float64')
            deviation_img = image.std(axis=(0, 1), dtype='float64')

            standardized_image = (image - mean_img) / deviation_img

            v_images.append(standardized_image.flatten())

            if csv_name != 'test.csv':
                v_labels.append(int(row['Class']))

    return np.array(v_images), np.array(v_labels)


def write_predictions(v_predicted_classes):
    with open('test.csv', 'r') as read_file:
        csv_content = pd.read_csv(read_file)

        v_images = csv_content['Image']

        data_to_write = pd.DataFrame({'Image': v_images, 'Class': v_predicted_classes})
        data_to_write.to_csv('sample_submission.csv', index=False)


class Statistics:
    def __init__(self, param_v_predicted_labels, param_v_given_labels, param_classes=96):
        self.v_predicted_labels = param_v_predicted_labels
        self.v_given_labels = param_v_given_labels
        self.classes = param_classes

        self.v_true_positive = []
        self.v_false_positive = []
        self.v_false_negative = []
        self.v_precision = np.zeros(self.classes)
        self.v_recall = np.zeros(self.classes)
        self.v_f1 = np.zeros(self.classes)

    def determine_metrics(self):
        for nr_class in range(self.classes):
            self.v_true_positive.append(
                np.sum((self.v_given_labels == nr_class) & (self.v_predicted_labels == nr_class)))
            self.v_false_positive.append(
                np.sum((self.v_given_labels != nr_class) & (self.v_predicted_labels == nr_class)))
            self.v_false_negative.append(
                np.sum((self.v_predicted_labels != nr_class) & (self.v_given_labels == nr_class)))

            if self.v_true_positive[nr_class] + self.v_false_positive[nr_class] > 0:
                self.v_precision[nr_class] = (self.v_true_positive[nr_class] / (
                        self.v_true_positive[nr_class] + self.v_false_positive[nr_class]))

            if self.v_true_positive[nr_class] + self.v_false_negative[nr_class] > 0:
                self.v_recall[nr_class] = (self.v_true_positive[nr_class] / (
                        self.v_true_positive[nr_class] + self.v_false_negative[nr_class]))

            if self.v_precision[nr_class] + self.v_recall[nr_class] > 0:
                self.v_f1[nr_class] = 2 * (self.v_precision[nr_class] * self.v_recall[nr_class]) / (
                        self.v_precision[nr_class] + self.v_recall[nr_class])

    def calculate_accuracy(self):
        return np.mean(self.v_predicted_labels == self.v_given_labels)

    def print_metrics_statistics(self):
        self.determine_metrics()

        print('\nStatistics for each class:\n')
        print('\t\tClass\t\tTrue Positives\t\tFalse Positives\t\tFalse Negatives\t\t\tPrecision\t\tRecall\t\tF1')
        print(
            '\t\t----------------------------------------------------------------------------------------------------------------')

        for nr_class in range(self.classes):
            str_nr_class = str(nr_class).ljust(20)
            str_true_positives = str(self.v_true_positive[nr_class]).ljust(21)
            str_false_positives = str(self.v_false_positive[nr_class]).ljust(20)
            str_false_negatives = str(self.v_false_negative[nr_class]).ljust(21)
            str_precision = '{:.3f}'.format(self.v_precision[nr_class]).ljust(13)
            str_recall = '{:.3f}'.format(self.v_recall[nr_class]).ljust(11)
            str_f1 = '{:.3f}'.format(self.v_f1[nr_class]).ljust(6)

            print(
                f'\t\t{str_nr_class}{str_true_positives}{str_false_positives}{str_false_negatives}{str_precision}{str_recall}{str_f1}')

    def show_confusion_matrix(self):
        confusion_matrix = np.zeros((self.classes, self.classes))

        for idx in range(len(self.v_given_labels)):
            confusion_matrix[int(self.v_given_labels[idx]), int(self.v_predicted_labels[idx])] += 1

        print("Associated confusion matrix:")
        for line in confusion_matrix:
            print(line)

        plt.imshow(confusion_matrix)

        plt.title("Confusion matrix")
        plt.colorbar(label="Nr. examples")

        plt.xlabel("Predicted classes")
        plt.ylabel("Actual classes")

        plt.show()


test_images, _ = get_dataset("test.csv")
train_images, train_labels = get_dataset("train.csv")
validation_images, validation_labels = get_dataset("val.csv")

knn_classifier = KNeighborsClassifier(n_neighbors=10, metric='cityblock')

knn_classifier.fit(train_images, train_labels)

predicted_validation_labels = knn_classifier.predict(validation_images)

statistics = Statistics(predicted_validation_labels, validation_labels)
print('\nTotal accuracy: {}%'.format(statistics.calculate_accuracy() * 100))

statistics.print_metrics_statistics()
statistics.show_confusion_matrix()

predicted_test_labels = knn_classifier.predict(test_images)
write_predictions(predicted_test_labels)
