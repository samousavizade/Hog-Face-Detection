import cv2
import numpy as np
import os
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import glob
from sklearn.metrics import accuracy_score
import logging
from sklearn.model_selection import train_test_split
from pickle import load, dump, HIGHEST_PROTOCOL
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, plot_precision_recall_curve, plot_roc_curve, roc_auc_score
import time
import gc
import tensorflow as tf


class Retriever:
    class RetrieveError(Exception):
        def __init__(self, message='Object file doesnt exist to retrieve ...'):
            super().__init__(message)

    def __init__(self, path):
        self.path = path

    def save(self, to_serialize_object):
        with open(self.path, 'wb') as file:
            dump(to_serialize_object, file, protocol=HIGHEST_PROTOCOL)

    def load(self):
        with open(self.path, 'rb') as file:
            return load(file)

    def exist_file(self):
        return os.path.isfile(self.path)

    def retrieve(self):
        if self.exist_file():
            return self.load()
        else:
            raise Retriever.RetrieveError()


def histogram_equalization(image, copy=False):
    if copy:
        image = image.copy()

    for i in range(3):
        image[:, :, i] = cv2.equalizeHist(image[:, :, i])

    return image


class HOGFaceDetector:
    class HOG:
        def __init__(self, winSize):
            self.winSize = winSize

            self.hog = cv2.HOGDescriptor(_winSize=winSize,
                                         _blockSize=(32, 32),
                                         _blockStride=(16, 16),
                                         _cellSize=(16, 16),
                                         _nbins=9,
                                         _derivAperture=1,
                                         _winSigma=-1.,
                                         _histogramNormType=0,
                                         _L2HysThreshold=0.2,
                                         _gammaCorrection=1,
                                         _nlevels=64,
                                         _signedGradient=True)

        def extract_features(self, image):
            fd = self.hog.compute(image).ravel()

            return fd

    def __init__(self,
                 positive_train_dir_path,
                 negative_train_dir_path,
                 winSize,
                 train_set_path,
                 validation_set_path,
                 test_set_path,
                 train_hog_features_path,
                 validation_hog_features_path,
                 test_hog_features_path,
                 svm_model_path,
                 y_predict_path):

        self.positive_train_dir_path = positive_train_dir_path
        self.negative_train_dir_path = negative_train_dir_path

        self.X_train, self.Y_train = None, None
        self.X_validation, self.Y_validation = None, None
        self.X_test, self.Y_test = None, None

        self.hog_object = None
        self.winSize = winSize

        self.train_HOG_features = None
        self.validation_HOG_features = None
        self.test_HOG_features = None

        self.Y_test_predicted = None

        self.scalar = None
        self.svm_kernel = ''

        self.classifier_model = None

        self.average_precision_recall_score = 0

        self.train_set_path = train_set_path
        self.validation_set_path = validation_set_path
        self.test_set_path = test_set_path
        self.train_hog_features_path = train_hog_features_path
        self.validation_hog_features_path = validation_hog_features_path
        self.test_hog_features_path = test_hog_features_path
        self.svm_model_path = svm_model_path
        self.y_predict_path = y_predict_path

    def initialize(self):
        # p: Positive / n: Negative

        retriever1 = Retriever(self.train_set_path)
        retriever2 = Retriever(self.validation_set_path)
        retriever3 = Retriever(self.test_set_path)

        try:
            self.X_train, self.Y_train = retriever1.retrieve()
            self.X_validation, self.Y_validation = retriever2.retrieve()
            self.X_test, self.Y_test = retriever3.retrieve()

        except:
            p_X, p_Y, positive_patches_size = self.retrieve_samples_of(self.positive_train_dir_path,
                                                                       label=1)

            n_X, n_Y, negative_patches_size = self.retrieve_samples_of(self.negative_train_dir_path,
                                                                       label=0)

            train_size = 20000 // 2
            validation_size = 2000 // 2
            test_size = 2000 // 2
            # total_size = train_size + validation_size + test_size

            p_X_train, p_Y_train, \
            p_X_validation, p_Y_validation, \
            p_X_test, p_Y_test = HOGFaceDetector.train_validation_test_split(p_X, p_Y,
                                                                             train_size,
                                                                             validation_size,
                                                                             test_size)

            n_X_train, n_Y_train, \
            n_X_validation, n_Y_validation, \
            n_X_test, n_Y_test = HOGFaceDetector.train_validation_test_split(n_X, n_Y,
                                                                             train_size,
                                                                             validation_size,
                                                                             test_size)

            self.X_train, self.Y_train = p_X_train + n_X_train, p_Y_train + n_Y_train
            retriever1.save([self.X_train, self.Y_train])

            self.X_validation, self.Y_validation = p_X_validation + n_X_validation, p_Y_validation + n_Y_validation
            retriever2.save([self.X_validation, self.Y_validation])

            self.X_test, self.Y_test = p_X_test + n_X_test, p_Y_test + n_Y_test
            retriever3.save([self.X_test, self.Y_test])

        logging.info('HOGFaceDetector object initialized ... ')

        return self

    @staticmethod
    def train_validation_test_split(p_X, p_Y, train_size, validation_size, test_size):
        p_X_train, p_X_test, p_Y_train, p_Y_test = train_test_split(p_X, p_Y,
                                                                    train_size=train_size + validation_size,
                                                                    test_size=test_size,
                                                                    random_state=1)

        p_X_train, p_X_validation, p_Y_train, p_Y_validation = train_test_split(p_X_train, p_Y_train,
                                                                                train_size=train_size,
                                                                                test_size=validation_size,
                                                                                random_state=1)

        return p_X_train, p_Y_train, p_X_validation, p_Y_validation, p_X_test, p_Y_test

    def retrieve_samples_of(self, path, label):
        directories = os.walk(path)
        base_path, class_names, _ = next(directories)
        Xs = []
        total_size = 0
        for index, class_name in enumerate(class_names):
            collection_path = base_path + '\\' + class_name
            pattern = collection_path + '\\*.jpg'

            image_paths = glob.glob(pattern)
            no_samples = len(image_paths)
            total_size += no_samples
            h, w = self.winSize
            class_collection = [cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR),
                                           dsize=(w, h),
                                           interpolation=cv2.INTER_CUBIC) for path in image_paths]

            Xs.append(class_collection)

        flat_Xs = [item for sublist in Xs for item in sublist]
        flat_Ys = [label for _ in range(total_size)]

        return flat_Xs, flat_Ys, total_size

    def extract_HOG_features(self):

        gc.collect()

        self.hog_object = HOGFaceDetector.HOG(self.winSize)

        retriever1 = Retriever(self.train_hog_features_path)
        retriever2 = Retriever(self.validation_hog_features_path)
        retriever3 = Retriever(self.test_hog_features_path)

        try:
            self.train_HOG_features = retriever1.retrieve()
            self.validation_HOG_features = retriever2.retrieve()
            self.test_HOG_features = retriever3.retrieve()

            print('Retrieved from file ...')
        except:

            features_ = [self.hog_object.extract_features(image) for image in self.X_train]
            self.train_HOG_features = features_
            retriever1.save(features_)

            features_ = [self.hog_object.extract_features(image) for image in self.X_validation]

            self.validation_HOG_features = features_
            retriever2.save(features_)

            features_ = [self.hog_object.extract_features(image) for image in self.X_test]
            self.test_HOG_features = features_
            retriever3.save(features_)

        # clear space

        del self.X_train
        del self.X_validation
        del self.X_test
        gc.collect()
        time.sleep(5)

        logging.info('feature descriptors of train and test set extracted ... ')

        return self

    @staticmethod
    def stack_descriptors(train_descriptors):
        train_descriptors = [en for sub_array in train_descriptors for en in sub_array]
        train_descriptors_vstack = np.vstack(train_descriptors)
        return train_descriptors_vstack

    def classify_svm(self, kernel):

        self.scalar = StandardScaler()
        self.train_HOG_features = self.scalar.fit_transform(self.train_HOG_features)
        self.validation_HOG_features = self.scalar.transform(self.validation_HOG_features)
        self.test_HOG_features = self.scalar.transform(self.test_HOG_features)

        gc.collect()

        retriever = Retriever(self.svm_model_path)

        try:
            self.classifier_model = retriever.load()
            print('Retrieved from file ...')
        except:

            # construct classifier_model model
            svc = SVC(kernel=kernel,
                      C=0.01,
                      max_iter=-1,
                      probability=True,
                      cache_size=600, )

            # estimate parameters
            # c_parameter = [i * 0.1 for i in range(3, 18)]
            # c_parameter = [1]
            # gamma_parameter = [10 ** i for i in range(-5, 1)]
            # gamma_parameter = [10 ** -4]

            # parameters = [{'kernel': [kernel], 'C': c_parameter, 'gamma': gamma_parameter}]

            # grid_search = GridSearchCV(scoring='accuracy',
            #                            estimator=svc,
            #                            param_grid=parameters)

            # fit model to estimate
            # grid_search = grid_search.fit(self.validation_HOG_features, self.Y_validation)

            self.classifier_model = svc
            # train classifier
            self.classifier_model.fit(self.train_HOG_features,
                                      self.Y_train)

            retriever.save(self.classifier_model)

        # self.classifier_model = grid_search.best_estimator_

        # predict test image labels
        self.Y_test_predicted = self.classifier_model.predict(self.test_HOG_features)

        logging.info('test set classified ')

        self.compute_metrics()
        logging.info('metrics computed and saved ')

        # clear space
        del self.train_HOG_features
        del self.validation_HOG_features
        del self.test_HOG_features
        gc.collect()
        time.sleep(5)

        return self

    def get_average_precision_recall_score(self):
        return self.average_precision_recall_score

    def get_predicted_labels(self):
        return self.Y_test_predicted

    def get_true_labels(self):
        return self.Y_test

    def compute_metrics(self):
        y_score = self.classifier_model.decision_function(self.test_HOG_features)

        av_precision_recall_score = average_precision_score(self.Y_test,
                                                            y_score)

        logging.info('Average precision-recall score: {0:0.2f}'.format(av_precision_recall_score))

        display = plot_precision_recall_curve(self.classifier_model,
                                              self.test_HOG_features,
                                              self.Y_test)

        display.ax_.set_title('Precision Recall Curve: AP = ' + str(av_precision_recall_score))
        plt.savefig('res2.jpg', format='jpg', dpi=400)
        logging.info('Average Precision-Recall Score: {0:0.2f}'.format(av_precision_recall_score))

        roc_area_under_curve = roc_auc_score(self.Y_test,
                                             y_score)

        display = plot_roc_curve(self.classifier_model,
                                 self.test_HOG_features,
                                 self.Y_test)

        display.ax_.set_title('ROC Curve: AUC = ' + str(roc_area_under_curve))
        plt.savefig('res1.jpg', format='jpg', dpi=400)
        logging.info('ROC Area Under Curve: {0:0.2f}'.format(roc_area_under_curve))


def add_padding(image, padding=100):
    h, w, _ = image.shape

    H = h + 2 * padding
    W = w + 2 * padding

    result = np.zeros((H, W, 3), dtype=np.uint8)
    result[padding:-padding, padding:-padding, :] = image

    return result


def FaceDetector(test_image, patch_h, patch_w, hog_face_detector: HOGFaceDetector, score_thresh=.995):
    padding = 50
    equalized_histogram_test_image = add_padding(test_image, padding=padding)
    test_image = add_padding(test_image, padding=padding)

    hog: HOGFaceDetector.HOG = hog_face_detector.hog_object
    scalar: StandardScaler = hog_face_detector.scalar
    classifier_model: SVC = hog_face_detector.classifier_model

    equalized_histogram_test_image = cv2.cvtColor(equalized_histogram_test_image, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = equalized_histogram_test_image.shape

    scales = [.05 * i for i in range(8, 21)]

    step = 1
    picked_patches = []
    for scale in scales:
        scaled_test_image = cv2.resize(equalized_histogram_test_image, dsize=(0, 0), fx=scale, fy=scale)
        H, W, _ = scaled_test_image.shape

        print('scale: ', str(scale))
        total_numbers = (H - patch_h) * (W - patch_w)

        patches = np.zeros((total_numbers, 6))
        index = 0
        for y in range(0, H - patch_h, step):
            for x in range(0, W - patch_w, step):
                part = scaled_test_image[y:y + patch_h, x:x + patch_w]
                feature_descriptors = hog.extract_features(part).reshape(1, -1)
                normal_feature_descriptors = scalar.transform(feature_descriptors)
                probability = classifier_model._predict_proba(normal_feature_descriptors)
                x1, y1, x2, y2 = x, y, x + patch_w, y + patch_h

                patches[index, :] = [probability[0, 1], y1, x1, y2, x2, scale]
                index += 1

        patches = patches[patches[:, 0] > score_thresh]
        picked_patches.append(patches)

    picked_patches = np.vstack(picked_patches)

    indices = tf.image.non_max_suppression(picked_patches[:, 1:5],
                                           picked_patches[:, 0],
                                           max_output_size=50,
                                           iou_threshold=.4,
                                           score_threshold=score_thresh).numpy()

    picked_patches = picked_patches[indices]

    indices = np.ones_like(indices)

    n = len(picked_patches)
    for i1 in range(n):
        for i2 in range(i1 - 1):
            p1 = picked_patches[i1]
            p2 = picked_patches[i2]
            y1_a, x1_a, y2_a, x2_a, scale_a = p1[1:]
            y1_a, x1_a, y2_a, x2_a = y1_a // scale_a, x1_a // scale_a, y2_a // scale_a, x2_a // scale_a

            y1_b, x1_b, y2_b, x2_b, scale_b = p2[1:]
            y1_b, x1_b, y2_b, x2_b = y1_b // scale_b, x1_b // scale_b, y2_b // scale_b, x2_b // scale_b

            S1 = (x2_a - x1_a) * (y2_a - y1_a)
            S1 = int(S1)

            S2 = (x2_b - x1_b) * (y2_b - y1_b)
            S2 = int(S2)

            overlap = max(0, min(x2_a, x2_b) - max(x1_a, x1_b)) * max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
            overlap = int(overlap)

            union = S1 + S2 - overlap
            # union = overlap
            if union == S1:
                indices[i1] = 0

            elif union == S2:
                indices[i2] = 0

    indices = np.where(indices == 1)

    final_patches = picked_patches[indices]

    for index, patch in enumerate(final_patches):
        probability, y1, x1, y2, x2, scale = patch
        y, x, h, w = y1, x1, y2 - y1, x2 - x1
        y, x, h, w = y // scale, x // scale, h // scale, w // scale
        y, x, h, w = int(y), int(x), int(h), int(w),
        y, x = y, x
        y1, x1, y2, x2 = y, x, y + h, x + w
        color = (0, 255, 0)
        test_image = cv2.rectangle(test_image, (x1, y1), (x2, y2), color=color, thickness=2)

        color = (255, 255, 255)
        test_image = cv2.putText(test_image,
                                 'P: ' + str(round(probability, 3)) + ', Scale: ' + str(scale),
                                 (x1, y1 - 5),
                                 cv2.FONT_HERSHEY_PLAIN,
                                 .7,
                                 color,
                                 thickness=1,
                                 lineType=2)

    cv2.imshow('f', test_image)
    cv2.waitKey(0)

    return test_image


def main():
    melli = cv2.imread('Melli.jpg')
    perspolis = cv2.imread('Persepolis.jpg')
    esteghlal = cv2.imread('Esteghlal.jpg')

    logging.root.setLevel(logging.INFO)

    positive_train_dir_path = r'lfw'
    negative_train_dir_path = r'256_ObjectCategories'

    # configurations
    kernel = 'linear'

    # construct bag of visual words object
    hog_face_detector = HOGFaceDetector(positive_train_dir_path,
                                        negative_train_dir_path,
                                        winSize=(128, 128),
                                        train_set_path='train_set.pkl',
                                        validation_set_path='validation_set.pkl',
                                        test_set_path='test_set.pkl',
                                        train_hog_features_path='train_hog_features.pkl',
                                        validation_hog_features_path='validation_hog_features.pkl',
                                        test_hog_features_path='test_hog_features.pkl',
                                        svm_model_path='svm.pkl',
                                        y_predict_path='y_predict.pkl')

    # fit model to predict labels
    test_predicted_labels = hog_face_detector. \
        initialize(). \
        extract_HOG_features(). \
        classify_svm(kernel=kernel). \
        get_predicted_labels()

    # true labels
    test_true_labels = hog_face_detector.get_true_labels()

    # compute accuracy based on test true labels and test predicted labels by model
    accuracy = accuracy_score(test_true_labels, test_predicted_labels)
    print('accuracy: ', accuracy)

    h, w = hog_face_detector.winSize

    test_image1_faces = FaceDetector(perspolis, h, w, hog_face_detector)
    cv2.imwrite('res4.jpg', test_image1_faces)

    test_image1_faces = FaceDetector(esteghlal, h, w, hog_face_detector)
    cv2.imwrite('res5.jpg', test_image1_faces)

    test_image1_faces = FaceDetector(melli, h, w, hog_face_detector)
    cv2.imwrite('res6.jpg', test_image1_faces)


if __name__ == '__main__':
    main()
