#!/usr/bin/env python

import os, sys
import cv2, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class mini_db(object):
    """docstring for mini_db."""
    def __init__(self, image_paths, image_shape, labels=None, batch_size=16):
        super(mini_db, self).__init__()
        self.image_paths = image_paths
        self.labels      = labels
        num_images       = len(self.image_paths)
        self.num_batches = num_images / batch_size

        # Create a queue that will contain image paths and their indices
        self.path_queue = tf.FIFOQueue(capacity=num_images,
                                       dtypes=[tf.int32, tf.string],
                                       name='path_queue')

        # Enqueue all image paths, along with their indices
        indices = tf.range(num_images)
        self.enqueue_paths_op = self.path_queue.enqueue_many([indices, self.image_paths])

        # Close the path queue (no more additions)
        self.close_path_queue_op = self.path_queue.close()

        (idx, img) = self.load_single(image_shape)

        processed_queue = tf.FIFOQueue(capacity=num_images,
                                       dtypes=[tf.int32, tf.float32],
                                       shapes=[(), image_shape],
                                       name='processed_queue')

        # Enqueue the processed image and path
        enqueue_processed_op = processed_queue.enqueue([idx, img])

        # Create a dequeue op that fetches a batch of processed images off the queue
        self.dequeue_op = processed_queue.dequeue_many(batch_size)

        # Create a queue runner to perform the processing operations in parallel
        self.queue_runner = tf.train.QueueRunner(processed_queue, [enqueue_processed_op])

    def load_single(self, image_shape):
        # Dequeue a single image path
        idx, image_path = self.path_queue.dequeue()
        # Read the file
        file_data = tf.read_file(image_path)
        img = tf.image.decode_jpeg(file_data, channels=image_shape[2])
        img = tf.image.resize_images(img, [image_shape[0], image_shape[1]])

        return (idx, img)

    def get(self, session):
        '''
        Get a single batch of images along with their indices. If a set of labels were provided,
        the corresponding labels are returned instead of the indices.
        '''
        (indices, images) = session.run(self.dequeue_op)
        if self.labels is not None:
            labels = [self.labels[idx] for idx in indices]
            return (labels, images)
        return (indices, images)

    def start(self, session, coordinator):
        '''Start the processing worker threads.'''
        # Queue all paths
        session.run(self.enqueue_paths_op)
        # Close the path queue
        session.run(self.close_path_queue_op)
        # Start the queue runner and return the created threads
        return self.queue_runner.create_threads(session, coord=coordinator, start=True)

    def batches(self, session):
        '''Yield a batch until no more images are left.'''
        for _ in xrange(self.num_batches):
            yield self.get(session=session)


if __name__ == '__main__':
    image_paths = [os.path.join('test_images', f) for f in os.listdir('test_images') if os.path.splitext(os.path.basename(f))[-1] == '.jpg']
    imdb = mini_db(image_paths, [224,224,3])

    with tf.Session() as sess:
        coordinator = tf.train.Coordinator()
        threads = imdb.start(sess, coordinator)
        for _ in xrange(imdb.num_batches):
            (indices, images) = imdb.get(session=sess)
            print images.shape
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)
