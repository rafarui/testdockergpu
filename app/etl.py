import luigi
import numpy as np

from pycarol.pipeline import inherit_list
from pycarol.pipeline.task.kubernetestask import EasyKubernetesTask as Task

import time

luigi.auto_namespace(scope='teste')


Task.is_cloud_target = True
Task.DOCKER_IMAGE = 'gcr.io/labs-ai-apps-qa/test_rafael/test'

class Task4(Task):
    def easy_run(self, input):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return [np.random.random() for _ in range(1000)]


@inherit_list(
    Task4,
)
class Task3(Task):
    def easy_run(self, input):
        time.sleep(10)

        return "asadsda"


class Task2(Task):
    def easy_run(self, input):
        return [np.random.random() for _ in range(1000)]


class Task1(Task):
    def easy_run(self, input):
        return [np.random.random() for _ in range(1000)]


class Task0(Task):
    def easy_run(self, input):
        return


@inherit_list(
    Task0,
    Task1,
    Task2,
    Task3
)
class MainTask(Task):

    def easy_run(self, input):
        return [np.random.random() for _ in range(1000)]