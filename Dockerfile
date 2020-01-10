# TODO: Smaller image
FROM seunglab/seamless:nkem-minnie-fine

RUN apt-get update
RUN pip install artificery
RUN pip install scikit-build
RUN pip install cloud-volume==0.61.0 google-api-core==1.14.2
RUN pip install pillow
RUN pip install task-queue

WORKDIR /workspace
ADD . /workspace
