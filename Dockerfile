# TODO: Smaller image
FROM gcr.io/zetta-aibs-mouse-001/seamless-fine-downsample-coarse

RUN apt-get update
RUN pip install artificery
# RUN pip install scikit-build
# RUN pip install cloud-volume
# RUN pip install pillow
# RUN pip install task-queue

WORKDIR /workspace
ADD . /workspace
