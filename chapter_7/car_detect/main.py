import cv2
import numpy as np

BOW_NUM_TRAINING_SAMPLES_PER_CLASS = 10
SVM_NUM_TRAINING_SAMPLES_PER_CLASS = 490

sift = cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDTREE = 1
index_param = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_param = {}
flann = cv2.FlannBasedMatcher(index_param, search_param)

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
bow_extractor = cv2.BOWImgDescriptorExtractor(sift, flann)


def get_pos_and_neg_path(i):
    pos_path = 'CarData/pos-%d.pgm' % (i+1)
    neg_path = 'CarData/neg-%d.pgm' % (i + 1)
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


for i in range(SVM_NUM_TRAINING_SAMPLES_PER_CLASS):
    pos_path, neg_path = get_pos_and_neg_path(i)
    add_data(pos_path, 1)
    add_data(neg_path, -1)

svm = cv2.ml.SVM_create()
svm.train(np.array(training_data), cv2.ml.ROW_SAMPLE,
          np.array(training_labels))

for test_img_path in ['cars_test/00010.jpg',
                      'cars_test/00009.jpg',
                      'cars_test/00003.jpg',
                      'cars_test/00004.jpg',
                      'cars_test/00005.jpg',
                      'cars_test/00006.jpg',
                      'cars_test/00007.jpg',
                      'cars_test/00008.jpg',
                      '../../images/car.jpg',
                      # '../../images/haying.jpg',
                      # '../../images/statue.jpg',
                      '../../images/woodcutters.jpg']:
    img = cv2.imread(test_img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    descriptor = extract_bow_descriptor(gray_img)
    prediction = svm.predict(descriptor)
    if prediction[1][0][0] == 1.0:
        text = 'car'
        color = [0, 255, 0]
    else:
        text = 'not car'
        color = [0, 0, 255]

    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.imshow(test_img_path, img)

cv2.waitKey()

