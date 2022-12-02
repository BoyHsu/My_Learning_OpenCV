import os.path

import cv2
import numpy as np
from non_max_suppression import non_max_suppression_fast as nms

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 100

SVM_SCORE_THRESHOLD = 1.8
NMS_OVERLAP_THRESHOLD = 0.15

sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_param = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_param = {}
flann = cv2.FlannBasedMatcher(index_param, search_param)

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)


def get_pos_and_neg_path(i):
    path = 'CarData/TrainImages'
    pos_path = '%s/pos-%d.pgm' % (path, (i + 1))
    neg_path = '%s/neg-%d.pgm' % (path, (i + 1))
    return pos_path, neg_path


def add_sample(path):
    if path is not  None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            keypoints, descriptors = sift.detectAndCompute(img, None)
            if descriptors is not None:
                bow_kmeans_trainer.add(descriptors)


for i in range(BOW_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_path(i)
    add_sample(pos_path)
    add_sample(neg_path)

voc = bow_kmeans_trainer.cluster()
bow_extractor.setVocabulary(voc)


def extract_bow_descriptor(img):
    features = sift.detect(img)
    return bow_extractor.compute(img, features)


training_data = []
training_labels = []


def add_data(path, label):
    if path is not None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            des = extract_bow_descriptor(img)
            if des is not None:
                training_data.extend(des)
                training_labels.append(label)


svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(50)

for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_path(i)
    add_data(pos_path, 1)
    add_data(neg_path, -1)

svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
          np.array(training_labels))

# file_my_svm = 'my_svm.xml'
# if os.path.exists(file_my_svm):
#     svm.load(file_my_svm)
# else:
#     for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
#         pos_path, neg_path = get_pos_and_neg_path(i)
#         add_data(pos_path, 1)
#         add_data(neg_path, -1)
#
#     svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
#               np.array(training_labels))
#     svm.save(file_my_svm)


def pyramid(img, scale_factor=1.25, min_size=(200, 80), max_size=(600, 600)):
    h, w = img.shape
    min_w, min_h = min_size
    max_w, max_h = max_size
    while w >= min_w and h >= min_h:
        if w <= max_w and h <= max_h:
            yield img
        w /= scale_factor
        h /= scale_factor
        img = cv2.resize(img, (int(w), int(h)),
                         interpolation=cv2.INTER_AREA)


def sliding_window(img, step=20, window_size=(100, 40)):
    img_h, img_w = img.shape
    window_w, window_h = window_size
    for y in range(0, img_w, step):
        for x in range(0, img_h, step):
            roi = img[y:y+window_h, x:x+window_w]
            roi_h, roi_w = roi.shape
            if roi_w == window_w and roi_h == window_h:
                yield x, y, roi


# for test_img_path in ['cars_test/00010.jpg',
#                       'cars_test/00009.jpg',
#                       'cars_test/00003.jpg',
#                       'cars_test/00004.jpg',
#                       'cars_test/00005.jpg',
#                       'cars_test/00006.jpg',
#                       'cars_test/00007.jpg',
#                       'cars_test/00008.jpg',
#                       '../../images/car.jpg',
#                       '../../images/haying.jpg',
#                       '../../images/statue.jpg',
#                       '../../images/woodcutters.jpg']:
for test_img_path in ['CarData/TestImages/test-0.pgm',
                      'CarData/TestImages/test-1.pgm',
                      '../../images/car.jpg',
                      '../../images/haying.jpg',
                      '../../images/statue.jpg',
                      '../../images/woodcutters.jpg']:
    img = cv2.imread(test_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pos_rects = []
    for resized in pyramid(gray_img):
        for x, y, roi in sliding_window(resized):
            descriptor = extract_bow_descriptor(roi)
            if descriptor is None:
                continue
            prediction = svm.predict(descriptor)
            if prediction[1][0][0] == 1.0:
                raw_prediction = svm.predict(
                    descriptor,
                    flags=cv2.ml.STAT_MODEL_RAW_OUTPUT
                )
                score = -raw_prediction[1][0][0]
                print(score)
                if score > SVM_SCORE_THRESHOLD:
                    h, w = roi.shape
                    scale = gray_img.shape[0] / \
                        float(resized.shape[0])
                    pos_rects.append([
                        int(x * scale),
                        int(y * scale),
                        int((x + w) * scale),
                        int((y + h) * scale),
                        score
                    ])

    pos_rects = nms(np.array(pos_rects), NMS_OVERLAP_THRESHOLD)
    print(len(pos_rects))
    for x0, y0, x1, y1, score in pos_rects:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)
        text = '%.2f' % score
        cv2.putText(img, text, (int(x0), int(y0)-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(test_img_path, img)

cv2.waitKey()

