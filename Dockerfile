from sergiypopo/myd

RUN apt-get update
RUN pip3 install artificery
RUN pip3 install scikit-build    
RUN pip3 install cloud-volume 

workdir /workspace
add . /workspace

