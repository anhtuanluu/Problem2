FROM python:3.7.13-slim

# RUN mkdir -p /usr/local/java

WORKDIR /bai2
COPY requirements.txt /bai2

RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y openjdk-11-jdk

# ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
# RUN export JAVA_HOME

COPY . /bai2
CMD ["python", "./app.py"]
# CMD ["bash"]