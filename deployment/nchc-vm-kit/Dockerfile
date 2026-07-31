# Stable CUDA-Q 0.15.0, CUDA 13, linux/amd64 manifest digest from NGC.
FROM nvcr.io/nvidia/quantum/cuda-quantum:cu13-0.15.0@sha256:cebc4669e40cb365d05d8084d5cf072695bca8b362e4ff46ea0dd4a65bfef15f

ARG OPENCODE_VERSION=1.18.1

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends acl ca-certificates curl git jq openssl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-course.txt /tmp/requirements-course.txt
RUN python -m pip install --no-cache-dir --break-system-packages -r /tmp/requirements-course.txt \
    && mkdir -p /opt/nchc-cudaq-course \
    && python -m pip freeze > /opt/nchc-cudaq-course/resolved-requirements.txt

USER cudaq
ENV PATH="/home/cudaq/.opencode/bin:${PATH}"
RUN curl -fsSL https://opencode.ai/install \
    | bash -s -- --version "${OPENCODE_VERSION}" --no-modify-path \
    && opencode --version

COPY --chown=cudaq:cudaq smoke-test-cudaq.py /opt/nchc-cudaq-course/smoke-test-cudaq.py
COPY --chown=cudaq:cudaq validate-course-notebooks.py /opt/nchc-cudaq-course/validate-course-notebooks.py

WORKDIR /workspace/cudaq-agentic-coding
EXPOSE 8888

# Remove the upstream login-shell entrypoint so docker run/Compose commands behave normally.
ENTRYPOINT []
CMD ["bash", "-l"]
