#!/usr/local/bin/python
import os
import luigi


from app.etl import MainTask


if __name__=="__main__":

    a = luigi.build([MainTask()], local_scheduler=True, workers=1, detailed_summary=True)

    print(a.status.name=='SUCCESS_WOTH_RETRY')
    print(a.status.name=='SUCCESS')