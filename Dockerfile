# VERSION           1

# Build
FROM tiagopeixoto/graph-tool:latest as graphtool
MAINTAINER Andrew Mellor <mellor91@hotmail.co.uk>

RUN pacman -S python-networkx python-matplotlib python-pip --noconfirm --needed
RUN pip install ipyparallel

ENV PYTHONIOENCODING=utf8
ENV PATH="/annotated:${PATH}"
ENV PYTHONPATH="/annotated:${PYTHONPATH}"

ENTRYPOINT jupyter notebook --ip 0.0.0.0 && bash