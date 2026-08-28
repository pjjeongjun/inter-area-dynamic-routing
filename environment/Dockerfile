# hash:sha256:fbb8cf42108f3b9842523967c1691f8152553d4c9ee04a618f990b49d0713330
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/swdb:Python3.12code-server4.133bedrock-auth0.5.0

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN conda install -y \
        jupyterlab==4.6.3 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    aind-data-access-api==1.10.0 \
    hdmf-zarr==0.13.0 \
    matplotlib==3.11.1 \
    numpy==2.5.2 \
    pandas==3.0.5 \
    pynwb==4.1.0 \
    scikit-learn \
    scipy==1.18.1 \
    seaborn==0.13.2
