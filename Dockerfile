#FROM nvcr.io/nvidia/pytorch:21.04-py3
#
#ARG DEBIAN_FRONTEND=noninteractive
#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#RUN pip uninstall opencv-contrib-python
#RUN pip uninstall opencv-python
#RUN pip uninstall opencv-python-headless
#
#
#RUN pip install opencv-contrib-python==4.5.5.62
#
#RUN pip uninstall -y scikit-image
#RUN pip install matplotlib scikit-learn scikit-image yacs joblib natsort h5py tqdm
#RUN pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips


FROM dpir-base:latest

#RUN pip install hdf5storage

COPY . /opt/dpir
WORKDIR /opt/dpir

#RUN python setup.py develop --no_cuda_ext
#CMD sleep infinity
CMD python main_dpir_deblur.py

