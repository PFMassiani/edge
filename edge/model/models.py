class Model:
    def __init__(self, space):
        self.space = space

    def update(self):
        raise NotImplementedError

    def query(self, point):
        raise NotImplementedError

    def __call__(self, point):
        return self.query(point)
