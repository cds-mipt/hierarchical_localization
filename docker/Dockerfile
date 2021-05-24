FROM colmap/colmap:latest
RUN apt-get update -y
RUN apt-get install python3 python3-pip unzip wget -y
RUN pip3 install --upgrade pip
RUN pip3 install torch
RUN pip3 install numpy
RUN pip3 install opencv-python
RUN pip3 install tqdm
RUN pip3 install matplotlib
RUN pip3 install scipy
RUN pip3 install h5py
RUN pip3 install jupyterlab notebook
RUN mkdir -p /home
WORKDIR /home
RUN pip3 install git+https://github.com/mihaidusmanu/pycolmap

# Installing g2o python
WORKDIR /home
RUN git clone https://github.com/uoip/g2opy.git
WORKDIR /home/g2opy
RUN mkdir build
WORKDIR /home/g2opy/build
RUN cmake -DPYBIND11_PYTHON_VERSION=3.6 ..
RUN make -j8
WORKDIR /home/g2opy
RUN python3 setup.py install