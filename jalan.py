import argparse
import os
from datetime import datetime
from multiprocessing import Process
from threading import Thread
from time import sleep

import cv2
import detect_and_align
import numpy as np
import tensorflow as tf
from base import engine
from models.log_person_counter import LogPersonCounterM
from models.log_unknown import LogUnknownM
from models.sub_device import SubDeviceM
from models.users import UserM
from scipy import misc
from sklearn.metrics.pairwise import pairwise_distances
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

tf = tf.compat.v1
tf.disable_v2_behavior()

Session = sessionmaker(bind=engine)


class IdData:

    def __init__(self, id_folder, mtcnn, sess, embeddings, images_placeholder,
                 phase_train_placeholder, distance_treshold):
        print('Loading known identities: ')
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []

        image_paths = []
        ids = os.listdir(os.path.expanduser(id_folder))
        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)
            image_paths += [os.path.join(id_dir, img) for img in os.listdir(id_dir)]

        print('Found %d images in id folder' % len(image_paths))
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        if len(id_image_paths) < 5:
            self.print_distance_table(id_image_paths)

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = misc.imread(os.path.expanduser(image_path), mode='RGB')
            face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                print("Warning: Found multiple faces in id image: %s" % image_path +
                      "\nMake sure to only have one face in the id images. " +
                      "If that's the case then it's a false positive detection and" +
                      " you can solve it by increasing the thresolds of the cascade network")
            aligned_images = aligned_images + face_patches
            id_image_paths += [image_path] * len(face_patches)
            self.id_names += [image_path.split('/')[-2]] * len(face_patches)

        return np.stack(aligned_images), id_image_paths

    def print_distance_table(self, id_image_paths):
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split('/')[-1] for path in id_image_paths]
        print('Distance matrix:\n{:20}'.format(''))
        [print('{:20}'.format(name), end='') for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print('\n{:20}'.format(path), end='')
            for distance in distance_row:
                print('{:20}'.format('%0.3f' % distance), end='')

    def find_matching_ids(self, embs):
        matching_ids = []
        matching_distances = []
        distance_matrix = pairwise_distances(embs, self.embeddings)
        for distance_row in distance_matrix:
            min_index = np.argmin(distance_row)
            if distance_row[min_index] < self.distance_treshold:
                matching_ids.append(self.id_names[min_index])
                matching_distances.append(distance_row[min_index])
            else:
                matching_ids.append(None)
                matching_distances.append(None)
        return matching_ids, matching_distances


def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print('Loading model filename: %s' % model_exp)
        with tf.gfile.GFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        raise ValueError('Specify model file, not directory!')


def cek_stat(idna, delay=0):
    sess = Session()
    sess.connection()
    subna = sess.query(SubDeviceM).filter_by(id=idna).first()
    # print(subna.id, subna.detect_stat)
    sess.close()
    sleep(delay)
    return subna


def main(args, data_cam):
    channel = "rtsp://{}:554/cam/realmonitor?channel={}&subtype=0".format(data_cam['host'], data_cam['channel'])
    cam_id = data_cam['id']
    while 1:
        stat = cek_stat(cam_id, 5)
        print(1)
        if stat.detect_stat:
            print(2)
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    cap = cv2.VideoCapture(channel)
                    if cap is None or not cap.isOpened():
                        print("")
                        print('Warning: unable to open video source: ', channel)
                    else:
                        mtcnn = detect_and_align.create_mtcnn(sess, None)
                        load_model(args.model)
                        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                        # Load anchor IDs
                        id_data = IdData(args.id_folder[0], mtcnn, sess, embeddings, images_placeholder,
                                         phase_train_placeholder, args.threshold)
                        while 1:
                            cek = cek_stat(cam_id)
                            if cek.detect_stat:
                                _, frame = cap.read()
                                face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(frame,
                                                                                                               mtcnn)
                                if len(face_patches) > 0:
                                    face_patches = np.stack(face_patches)
                                    feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                                    embs = sess.run(embeddings, feed_dict=feed_dict)

                                    matching_ids, matching_distances = id_data.find_matching_ids(embs)

                                    for bb, landmark, matching_id, dist in zip(
                                            padded_bounding_boxes, landmarks, matching_ids, matching_distances):
                                        if matching_id is None:
                                            name = datetime.now().strftime('%d%m%Y_%H%M%S_%f')
                                            f_name = name + ".jpeg"
                                            print('Unknown! Couldn\'t fint match.')
                                            if not os.path.exists('../api/_api/static/unknown/{}'.format(cam_id)):
                                                os.makedirs('../api/_api/static/unknown/{}'.format(cam_id))
                                            cv2.imwrite('../api/_api/static/unknown/{}/{}'.format(cam_id, f_name),
                                                        frame)
                                            session_unknown = Session()
                                            session_unknown.connection()
                                            try:
                                                session_unknown.add(LogUnknownM(f_name, cam_id))
                                                session_unknown.commit()
                                                print("Succesfully save {} data".format(f_name))
                                            except Exception as e:
                                                print(e)
                                                session_unknown.rollback()
                                                print("Failed to save")
                                            finally:
                                                session_unknown.close()
                                        else:
                                            print('Hi %s! Distance: %1.4f' % (matching_id, dist), end='\r')
                                            session_known = Session()
                                            session_known.connection()
                                            try:
                                                idna = session_known.query(UserM).filter_by(name=matching_id).first()
                                                print(idna).json()
                                                ada = session_known.query(LogPersonCounterM).filter_by(
                                                    user_id=idna).first()
                                                print(ada.json())
                                                last = ada.add_time.date()
                                                now = datetime.now().date()
                                                print(last, now)
                                                if now > last:
                                                    session_known.add(LogPersonCounterM(idna, cam_id))
                                                    session_known.commit()
                                            except Exception as e:
                                                print(e)
                                                session_known.rollback()
                                                print("Failed to save")
                                            finally:
                                                session_known.close()
                                else:
                                    print('Couldn\'t find a face')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str, help='Path to model protobuf (.pb) file')
    parser.add_argument('id_folder', type=str, nargs='+', help='Folder containing ID folders')
    parser.add_argument('-t', '--threshold', type=float, help='Distance threshold defining an id match', default=1.2)

    session = Session()

    list_camera = session.query(SubDeviceM).all()

    for kam in list_camera:
        isi = kam.json()
        Process(target=main, args=(parser.parse_args(), isi)).start()
