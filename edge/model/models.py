class Model:
    def __init__(self, space):
        self.space = space

    def update(self):
        raise NotImplementedError

    def query(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.query(*args, **kwargs)
