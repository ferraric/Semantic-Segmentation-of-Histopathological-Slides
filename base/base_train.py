from comet_ml import Experiment
import copy


class BaseTrain:
    def __init__(self, model, data, config, comet_logger):
        self.model = model
        self.config = config
        self.data = data

        assert (
            "num_epochs" in self.config
        ), "You need to define the parameter 'num_epochs' in your config file."
        assert (
            "num_iter_per_epoch" in self.config
        ), "You need to define the parameter 'num_iter_per_epoch' in your config file."
        assert (
            "test_every_x_iter" in self.config
        ), "You need to define the parameter 'test_every_x_iter' in your config file."

        # Add hyperparameters of config file to Comet Logger
        comet_logger.log_parameters(config)
        # TODO Add model graph to comet logger, currently no way known way to get model from TF 2.0
        # comet_logger.set_model_graph(model)

    def train(self):
        for cur_epoch in range(self.config.num_epochs):
            self.current_epoch = cur_epoch
            self.train_epoch()

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        """
        raise NotImplementedError

    def test_step(self):
        """implement the logic of the test step"""
        raise NotImplementedError

    def save_model(self):
        """implement the logic of saving the model and checkpoints"""
        raise NotImplementedError
