FROM python:3.9.13

WORKDIR /bai2
COPY requirements.txt /bai2

RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt


COPY . /bai2
CMD ["python", "./app.py"]