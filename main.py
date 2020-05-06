import os
import numpy as np
from pycarol.pipeline import inherit_list
from pycarol.pipeline.task.kubernetestask import EasyKubernetesTask as Task

import time

Task.DOCKER_IMAGE = os.environ['DOCKER_IMAGE']

class Task4(Task):
    def easy_run(self, input):
        print('task ', self.__class__)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return [np.random.random() for _ in range(1000)]


@inherit_list(
    Task4,
)
class Task3(Task):
    def easy_run(self, input):
        print('task ', self.__class__)
        time.sleep(10)

        return "asadsda"


class Task2(Task):
    def easy_run(self, input):
        print('task ',self.__class__)
        return [np.random.random() for _ in range(1000)]


class Task1(Task):
    def easy_run(self, input):
        print('task ', self.__class__)
        return [np.random.random() for _ in range(1000)]


class Task0(Task):
    def easy_run(self, input):
        print('task ', self.__class__)
        return


@inherit_list(
    Task0,
    Task1,
    Task2,
    Task3
)
class MainTask(Task):

    def easy_run(self, input):
        print('task ', self.__class__)
        return [np.random.random() for _ in range(1000)]
