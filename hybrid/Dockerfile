FROM tensorflow/tensorflow:latest-gpu
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y wget
RUN apt-get install -y git
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata
RUN apt-get -y install libopencv-dev python3-opencv
RUN pip install opencv-contrib-python
RUN pip install pillow
RUN pip install keras_applications
RUN wget https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt
RUN git clone https://github.com/GKalliatakis/Keras-VGG16-places365
RUN pip install keras python-dotenv kafka-python redis mongoengine
WORKDIR Keras-VGG16-places365
EXPOSE 8000
COPY main.py .
CMD ["python", "main.py"]

