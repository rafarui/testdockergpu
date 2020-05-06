FROM python:3.6

RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app/
RUN pip install -r requirements.txt

ARG GITTOKEN
ARG USER='totvslabs'

ARG REPO='pycarol'
ARG BRANCH='dev-k8s-luigi'
RUN git clone -b$BRANCH  --depth 1 https://${GITTOKEN}@github.com/$USER/$REPO.git


ADD . /app

RUN chmod a+x run.sh

CMD ["/bin/bash", "run.sh"]