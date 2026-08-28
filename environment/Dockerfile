# hash:sha256:ae64db5b292b2ad8a0938ef7de7e981d375d5f78910c57226f6c054864898f01
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
    nbstripout \
    numpy==2.5.2 \
    pandas==3.0.5 \
    pynwb==4.1.0 \
    scipy==1.18.1
