FROM nvcr.io/nvidia/pytorch:21.11-py3

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev

RUN pip3 install atomai

RUN mkdir /workdir
WORKDIR /workdir

RUN mkdir /elit
ADD src /elit/src
ADD bin /elit/bin

ENV PYTHONPATH "${PYTHONPATH}:/elit/src"
ENV PATH "${PATH}:/elit/bin"
