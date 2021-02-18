FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04 AS devel
ARG NCCL_VERSION_OVERRIDE
ENV NCCL_VERSION ${NCCL_VERSION_OVERRIDE:-2.8.4}
ENV NCCL_PKG_VERSION ${NCCL_VERSION}-1+cuda11.2
RUN apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        vim gdb git wget tar python-dev python3-dev \
        zlib1g-dev lsb-release ca-certificates clang-format libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p /var/tmp && wget -q -nc --no-check-certificate -P /var/tmp http://repo.anaconda.com/miniconda/Miniconda3-4.7.12-Linux-x86_64.sh && \
    bash /var/tmp/Miniconda3-4.7.12-Linux-x86_64.sh -b -p /opt/conda && \
    /opt/conda/bin/conda init && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    /opt/conda/bin/conda clean -afy && \
    rm -rf /var/tmp/Miniconda3-4.7.12-Linux-x86_64.sh
ENV CPATH=/opt/conda/include:$CPATH \
    LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH \
    LIBRARY_PATH=/opt/conda/lib:$LIBRARY_PATH \
    PATH=/opt/conda/bin:$PATH \
    CONDA_PREFIX=/opt/conda \
    NCCL_LAUNCH_MODE=PARALLEL
RUN conda update -n base -c defaults conda && \
    conda install -c anaconda cmake=3.18.2 pip && \
    conda install -c conda-forge ucx libhwloc=2.4.0 openmpi=4.1.0 openmpi-mpicc=4.1.0 mpi4py=3.0.3 && \
    rm -rf /opt/conda/include/nccl.h /opt/conda/lib/libnccl.so /opt/conda/include/google
ENV OMPI_MCA_plm_rsh_agent=sh \
    OMPI_MCA_opal_cuda_support=true
RUN echo alias python='/usr/bin/python3' >> /etc/bash.bashrc && \
    pip3 install numpy pandas sklearn ortools jupyter tensorflow==2.4.0
RUN mkdir -p /var/tmp && cd /var/tmp && git clone --depth=1 --branch branch-0.17 https://github.com/rapidsai/rmm.git rmm && cd - && \
    cd /var/tmp/rmm && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX && make -j && \
    cd /var/tmp/rmm && \
    cd build && make install && \
    rm -rf /var/tmp/rmm
RUN echo "deb http://cuda-repo/release-candidates/repos/nccl_r2.8_CUDA11.2/ubuntu2004/x86_64/"; > /etc/apt/sources.list.d/dgx.list && \
    echo "Switching NCCL to version ${NCCL_VERSION} using ${NCCL_PKG_VERSION}" && \
    apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        libnccl2=$NCCL_PKG_VERSION  \
        libnccl-dev=$NCCL_PKG_VERSION
COPY . HugeCTR
RUN cd HugeCTR && \
    git submodule update --init --recursive && \
    mkdir build && cd build &&\
    cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" \
         -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
         -DVAL_MODE=OFF -DENABLE_MULTINODES=ON -DENABLE_MPI=ON -DNCCL_A2A=ON -DUCX_INCLUDE_DIR=/usr/local/ucx/include/ -DUCX_LIBRARIES=/usr/local/ucx/lib/ .. && \
    make -j$(nproc) &&\
    mkdir /usr/local/hugectr &&\
    make install &&\
    chmod +x /usr/local/hugectr/bin/* &&\
    rm -rf HugeCTR

ENV PATH=/usr/local/hugectr/bin:$PATH