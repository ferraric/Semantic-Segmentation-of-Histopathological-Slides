import numpy as np
import os, sys
import tensorflow as tf

from base.base_train import BaseTrain
from tqdm import tqdm
from utils.dirs import list_files_in_directory


class ExampleSegmentationTrainer(BaseTrain):
    def __init__(self, model, data, config, comet_logger):
        super(ExampleSegmentationTrainer, self).__init__(
            model, data, config, comet_logger
        )
        self.optimizer = tf.keras.optimizers.Adam()
        self.comet_logger = comet_logger
        self.setup_metrics()

    def setup_metrics(self):
        """Define the metrics that we want to track"""
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.best_loss = sys.maxsize

        # Train Losses
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )

        # Test Losses
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="test_accuracy"
        )

    def train_epoch(self):
        """Loop over the iterations of one epoch and train, test, save the model"""
        loop = tqdm(range(self.config.num_iter_per_epoch))

        for _ in loop:
            # test after x iterations
            if self.optimizer.iterations % self.config.test_every_x_iter == 0:
                self.test_step()

            # optimize model
            self.train_step()

        # reset train loss and accuracy for next epoch
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()

    def train_step(self):
        """One Train step, i.e. optimize model"""
        # get data
        iterator = iter(self.data.train_data)
        inputs, labels = next(iterator)

        with self.comet_logger.train():
            # forward pass
            with tf.GradientTape() as tape:
                predictions = self.model(inputs)
                loss = self.loss_object(labels, predictions)

            # calculate gradients and apply them
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

            # update metrics
            self.train_loss(loss)
            self.train_accuracy(labels, predictions)
            self.comet_logger.log_metric("loss", loss, step=self.optimizer.iterations)
            self.comet_logger.log_metric(
                "average_loss", self.train_loss.result(), step=self.optimizer.iterations
            )
            self.comet_logger.log_metric(
                "average_accuracy",
                self.train_accuracy.result(),
                step=self.optimizer.iterations,
            )

    def test_step(self):
        # get data
        iterator = iter(self.data.test_data)
        inputs, labels = next(iterator)

        with self.comet_logger.test():
            predictions = self.model(inputs)
            loss = self.loss_object(labels, predictions)

            # if loss decreased, save model and delete previously saved models
            if loss < self.best_loss:
                self.best_loss = loss
                model_files = list_files_in_directory(self.config.checkpoint_dir)
                self.save_model()
                for f in model_files:
                    os.remove(f)

            # update metrics
            self.test_loss(loss)
            self.test_accuracy(labels, predictions)
            self.comet_logger.log_metric("loss", loss, step=self.optimizer.iterations)
            self.comet_logger.log_metric(
                "average_loss", self.test_loss.result(), step=self.optimizer.iterations
            )
            self.comet_logger.log_metric(
                "average_accuracy",
                self.test_accuracy.result(),
                step=self.optimizer.iterations,
            )

    def save_model(self):
        tf.saved_model.save(
            self.model,
            os.path.join(
                self.config.checkpoint_dir,
                "model_at_iter_" + str(self.optimizer.iterations),
            ),
        )
