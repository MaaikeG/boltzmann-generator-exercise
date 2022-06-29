import abc


class Conditioner:
    @abc.abstractmethod
    def __call__(self, *args):
        pass
