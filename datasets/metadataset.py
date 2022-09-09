import numpy as np


def raiseError(routine, parentclass):
    raise NotImplementedError("{} should be re-implemented in subclass of {}.".format(routine, parentclass))


class MetaDataset:
    def __init__(self):
        self.tasks = None

    def create_tasks(self):
        raiseError("create_tasks()", "MetaDataset")

    def sample_task(self, idx=None):
        if idx is None:
            idx = np.random.randint(0, len(self.tasks))
        return self.tasks[idx]

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return self.tasks[idx]


class MetaTask:

    def sample(self):
        raiseError("sample()", "MetaTask")

    def get_trainbatch(self):
        raiseError("get_trainbatch()", "MetaTask")

    def get_testbatch(self):
        raiseError("get_testbatch()", "MetaTask")
