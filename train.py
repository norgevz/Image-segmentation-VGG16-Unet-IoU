from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import tensorflow as tf
import lib
from tfmodel import model_input_fn

def main(args):

    for _ in range(lib._EPOCHS // lib._EPOCHS_PER_EVAL):
        tensors_to_log = {
            'cross_entropy': 'cross_entropy'
        }

        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)
        train_hooks = [logging_hook]

        # cfg = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
        run_config = tf.estimator.RunConfig(
            model_dir=lib.model_dir
        )

        # Create the Estimator
        model = tf.estimator.Estimator(
            model_fn=model_input_fn,
            model_dir=lib.model_dir,
            config=run_config,
            params={
                'n_classes': lib._NUM_CLASSES,
                'batch_size': lib._BATCH_SIZE
            }
        )

        tf.logging.info("Start Evaluation")
        model.train(
            input_fn=lambda: lib.input_fn_from_tfrecords(True, data_dir=lib.records_dir, batch_size=lib._BATCH_SIZE, epochs=lib._EPOCHS),
            hooks=train_hooks
        )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
