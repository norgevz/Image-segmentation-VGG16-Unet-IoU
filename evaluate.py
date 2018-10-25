from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import timeit

import tensorflow as tf
import numpy as np

import lib
import tfmodel


# implementation of mean_iou metric using confusion_matrix as a numpy array
def compute_mean_iou(total_cm):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = np.sum((denominator != 0).astype(float))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))

    ious = cm_diag / denominator

    print('Intersection over Union for each class:')
    for i, iou in enumerate(ious):
        print('    class {}: {:.4f}'.format(i, iou))

    # If the number of valid entries is 0 (no classes) we return 0.
    m_iou = np.where(
        num_valid_entries > 0,
        np.sum(ious) / num_valid_entries,
        0)
    m_iou = float(m_iou)
    print('mean Intersection over Union: {:.4f}'.format(float(m_iou)))


# implementation of accuracy metric using confusion_matrix as a numpy array
def compute_accuracy(total_cm):
  """Compute the accuracy via the confusion matrix."""
  denominator = total_cm.sum().astype(float)
  cm_diag_sum = np.diagonal(total_cm).sum().astype(float)

  # If the number of valid entries is 0 (no classes) we return 0.
  accuracy = np.where(
      denominator > 0,
      cm_diag_sum / denominator,
      0)
  accuracy = float(accuracy)
  print('Pixel Accuracy: {:.4f}'.format(float(accuracy)))


def main(unused_argv):

    features, labels = lib.input_fn(False, lib.records_dir, batch_size=1, epochs=1)
    model = tfmodel.model_input_fn(
        features,
        labels,
        tf.estimator.ModeKeys.EVAL,
        params={
            'n_classes': lib._NUM_CLASSES,
            'batch_size': 1
        })
    predictions = model.predictions

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(lib.model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)


        # Loop through the batches and store predictions and labels
        step = 1
        sum_cm = np.zeros((lib._NUM_CLASSES, lib._NUM_CLASSES), dtype=np.int32)
        start = timeit.default_timer()

        # accumulating confusion matrix over the eval dataset
        while True:
            try:
                pred = sess.run(predictions)
                sum_cm += pred['confusion_matrix']

                if not step % 100:
                    stop = timeit.default_timer()
                    tf.logging.info("current step = {} ({:.3f} sec)".format(step, stop - start))
                    start = timeit.default_timer()
                step += 1
            except tf.errors.OutOfRangeError:
                print("step --> " + str(step))
                break

        print("Confusion matrix created")
        compute_mean_iou(sum_cm)
        compute_accuracy(sum_cm)
        print("Done")



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)

