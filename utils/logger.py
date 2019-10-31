import tensorflow as tf
import os

def log_model_architecture_to(model, experiment, input_shape, summary_directory):
    model.build(input_shape)
    model_architecture_path = os.path.join(summary_directory, "model_architecture")
    with open(model_architecture_path, "w") as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + "\n"))
    print(model.summary())
    experiment.log_asset(model_architecture_path)