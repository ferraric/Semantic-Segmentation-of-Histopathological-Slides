class TrainValData:
    def __init__(self, train_data, val_data):
        self.train_data = train_data.dataset
        self.validation_data = val_data.dataset
