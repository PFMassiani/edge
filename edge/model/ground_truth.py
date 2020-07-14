class GroundTruth:
    """
    Base class for ground truths. A ground truth is an object that can be computed and from which we can get training
    examples (to fit a Model, for example). Ground truths may come from vibly: when necessary, the from_vibly_file
    method may be redefined.
    """
    def __init__(self):
        pass

    def compute(self):
        """
        Computes the ground truth. Typically computationally expensive.
        """
        raise NotImplementedError

    def get_training_examples(self):
        """
        Returns training examples to fit a model.
        :return: training examples, with labels
        """
        pass

    def from_vibly_file(self, vibly_file_path):
        """
        Load from a vibly ground truth.
        :param vibly_file_path: path to the vibly ground truth
        """
        raise NotImplementedError

    def save(self, save_path):
        """
        Save in a folder
        :param save_path: path where to save
        """
        raise NotImplementedError

    @staticmethod
    def load(load_path):
        """
        Load a ground truth from a file/folder as created by the GroundTruth.save method. Note that this method may fail
        if the save was made with an older version of the code.
        :param load_path: path from where to load
        """
        raise NotImplementedError