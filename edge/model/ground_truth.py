class GroundTruth:
    def __init__(self):
        pass

    def compute(self):
        raise NotImplementedError

    def get_training_examples(self):
        pass

    def from_vibly_file(self, vibly_file_path):
        raise NotImplementedError
