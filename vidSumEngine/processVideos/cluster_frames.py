import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil


def cluster_frames(input_dir):
    glob_dir = input_dir + '/*.jpg'
    print(glob_dir)

    for file in glob.glob(glob_dir):
        print("here")
        print(file)

    images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
    print(images)
    paths = [file for file in glob.glob(glob_dir)]
    images = np.array(np.float32(images).reshape(len(images), -1) / 255)

    model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    predictions = model.predict(images.reshape(-1, 224, 224, 3))
    pred_images = predictions.reshape(images.shape[0], -1)

    sil = []
    kl = []
    kmax = 10
    s_dict = {}

    for k in range(2, kmax + 1):
        kmeans2 = KMeans(n_clusters=k).fit(pred_images)
        labels = kmeans2.labels_
        score = silhouette_score(pred_images, labels, metric='euclidean')
        s_dict[score] = k

    max_k = max(s_dict)

    kmodel = KMeans(n_clusters=s_dict[max_k], random_state=728)
    kmodel.fit(pred_images)
    kpredictions = kmodel.predict(pred_images)
    if not os.path.exists('output'):
        os.mkdir('output')
    shutil.rmtree('output')
    for i in range(s_dict[max_k]):
        os.makedirs("/home/dikshakewat/Projects/VideoSumEngine/cluster_frames/cluster" + str(i))
    for i in range(len(paths)):
        shutil.copy2(paths[i], "/home/dikshakewat/Projects/VideoSumEngine/cluster_frames/cluster" + str(kpredictions[i]))
    # shutil.rmtree('/home/dikshakewat/Projects/VideoSumEngine/extracted_frames')


if __name__ == '__main__':
    cluster_frames('/home/dikshakewat/Projects/VideoSumEngine/extracted_frames')