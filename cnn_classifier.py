import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tfl
from keras import layers, callbacks
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import math

''' HYPERPARAMETERS '''
hyper_train_epochs = 200
hyper_dropout_rate = 0.3
hyper_learning_rate_init = 0.001
hyper_decay_rate = 0.3
hyper_kernel_reg = 0.035
hyper_patience = 30
hyper_batch_size = 48

gpus = tfl.config.list_physical_devices('GPU')
if gpus:
    try:
        tfl.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tfl.config.experimental.set_memory_growth(gpus[0], True)
        print('Connected to GPU:', gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print('No GPU available')


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
            image = Image.open(folder_name + row['Image'])

            image_resized = image.resize((64, 64))
            image_normalized = np.array(image_resized).astype('float32') / 255.0

            v_images.append(image_normalized)

            if csv_name != 'test.csv':
                v_labels.append(int(row['Class']))

    return np.array(v_images), np.array(v_labels)


def write_predictions(v_predicted_classes):
    with open('test.csv', 'r') as read_file:
        csv_content = pd.read_csv(read_file)

        v_images = csv_content['Image']

        data_to_write = pd.DataFrame({'Image': v_images, 'Class': v_predicted_classes})
        data_to_write.to_csv('sample_submission.csv', index=False)


def own_random_erasing(image, erase_probability=0.5, area_range=(0.02, 0.33), aspect_ratio_range=(0.3, 3)):
    if np.random.rand() > erase_probability:
        return image

    img_height, img_width, nr_channels = image.shape

    random_area = np.random.uniform(*area_range) * (img_height * img_width)
    random_aspect_ratio = np.random.uniform(*aspect_ratio_range)

    erase_height = min(int(round(np.sqrt(random_area * random_aspect_ratio))), img_height - 1)
    erase_width = min(int(round(np.sqrt(random_area / random_aspect_ratio))), img_width - 1)

    lower_bound_intact = np.random.randint(0, img_height - erase_height)
    left_bound_intact = np.random.randint(0, img_width - erase_width)

    image[lower_bound_intact:lower_bound_intact + erase_height,
        left_bound_intact:left_bound_intact + erase_width, :] = \
        (np.random.uniform(0, 1, (erase_height, erase_width, nr_channels)))

    return image


image_augmentation = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=own_random_erasing
)


class CNN_Classifier:
    def __init__(self, param_dropout_rate=0.25, param_kernel_reg=0.03, param_patience=25, param_learning_rate_init=0.001,
                 param_decay_rate=0.3, param_epochs=200, param_classes=96):
        self.cnn_model = tfl.keras.models.Sequential()

        self.cnn_model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu', input_shape=(64, 64, 3)))
        self.cnn_model.add(layers.BatchNormalization())

        self.cnn_model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.cnn_model.add(layers.BatchNormalization())
        self.cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.cnn_model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.cnn_model.add(layers.BatchNormalization())

        self.cnn_model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.cnn_model.add(layers.BatchNormalization())
        self.cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.cnn_model.add(layers.Dropout(rate=param_dropout_rate))

        self.cnn_model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.cnn_model.add(layers.BatchNormalization())

        self.cnn_model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.cnn_model.add(layers.BatchNormalization())
        self.cnn_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        self.cnn_model.add(layers.Dropout(rate=param_dropout_rate))

        self.cnn_model.add(layers.Flatten())

        self.cnn_model.add(layers.Dense(units=512, activation='relu', kernel_regularizer=tfl.keras.regularizers.l2(param_kernel_reg)))
        self.cnn_model.add(layers.BatchNormalization())

        self.cnn_model.add(layers.Dense(units=128, activation='relu', kernel_regularizer=tfl.keras.regularizers.l2(param_kernel_reg)))
        self.cnn_model.add(layers.BatchNormalization())

        self.cnn_model.add(layers.Dense(units=param_classes, activation='softmax'))

        self.cnn_model.compile(optimizer=tfl.keras.optimizers.Adam(weight_decay=0.0001),
                               loss=tfl.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

        self.callback_checkpoint = callbacks.ModelCheckpoint(filepath='saved_model.h5', monitor='val_accuracy',
                                                             save_best_only=True, mode='max', verbose=1)

        self.callback_early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=param_patience,
                                                               verbose=1, restore_best_weights=True)

        self.callback_cosine_annealing_lr = callbacks.LearningRateScheduler(
            lambda ep: param_learning_rate_init * param_decay_rate * (1 + math.cos((math.pi * ep) / param_epochs)))

    def train_model(self, param_train_images, param_train_labels, param_validation_images,
                    param_validation_labels, param_batch_size=48, param_epochs=200):
        self.cnn_model.summary()

        return self.cnn_model.fit(
            image_augmentation.flow(param_train_images, param_train_labels, batch_size=param_batch_size),
            steps_per_epoch=len(param_train_images) // param_batch_size,
            batch_size=param_batch_size, epochs=param_epochs,
            validation_data=(param_validation_images, param_validation_labels),
            callbacks=[self.callback_early_stopping, self.callback_cosine_annealing_lr, self.callback_checkpoint])

    def get_saved_model(self, file_name='saved_model.h5'):
        self.cnn_model = tfl.keras.models.load_model(file_name)

    def generate_predictions(self, param_v_images):
        return self.cnn_model.predict(param_v_images).argmax(axis=1)


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

            print(f'\t\t{str_nr_class}{str_true_positives}{str_false_positives}{str_false_negatives}{str_precision}{str_recall}{str_f1}')

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

print(train_images.shape)
print(train_labels.shape)

cnn_classifier = CNN_Classifier(param_dropout_rate=hyper_dropout_rate, param_kernel_reg=hyper_kernel_reg,
                                param_patience=hyper_patience, param_learning_rate_init=hyper_learning_rate_init,
                                param_decay_rate=hyper_decay_rate, param_epochs=hyper_train_epochs, param_classes=96)

model_history = cnn_classifier.train_model(train_images, train_labels, validation_images, validation_labels,
                                           param_batch_size=hyper_batch_size, param_epochs=hyper_train_epochs)

cnn_classifier.get_saved_model()

v_predicted_labels = cnn_classifier.generate_predictions(validation_images)

statistics = Statistics(v_predicted_labels, validation_labels)
print('\nTotal accuracy: {}%'.format(statistics.calculate_accuracy() * 100))

statistics.print_metrics_statistics()
statistics.show_confusion_matrix()

pred_test_labels = cnn_classifier.generate_predictions(test_images)
write_predictions(pred_test_labels)
