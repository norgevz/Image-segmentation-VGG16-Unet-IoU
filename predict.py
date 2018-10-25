import tensorflow as tf
import sys
import lib
import os
from tfmodel import model_input_fn


images_folder = 'data/images'
labels_folder = None
output_folder = 'output/'


def main(unused_argv):

    model = tf.estimator.Estimator(
        model_input_fn,
        model_dir=lib.model_dir,
        params={
            'n_classes': lib._NUM_CLASSES,
            'batch_size': 1
        }
    )

    image_files = [os.path.join(images_folder, file) for file in os.listdir(images_folder)]


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    predictions = model.predict(lambda: lib.input_fn_from_folder(image_files))
    for pred in predictions:
        print(pred)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]])