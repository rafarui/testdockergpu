FROM python:3.6

RUN mkdir /app
WORKDIR /app
RUN git clone https://github.com/rafarui/testdockergpu.git
WORKDIR /app/testdockergpu
RUN pip install -r requirements.txt

CMD ["python", "main.py"]