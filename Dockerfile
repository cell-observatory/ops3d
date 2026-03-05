# docker buildx build . --tag ghcr.io/cell-observatory/ops3d:main_torch_26_01 --build-arg BRANCH_NAME=$(git branch --show-current) --target torch_25_08 --progress=plain --no-cache-filter pip_install

# docker run --network host -u 1000 --privileged -v ~/.ssh:/sshkey -v ${PWD}:/workspace/ops3d --env PYTHONUNBUFFERED=1 --pull missing -t -i --rm -w /workspace/ops3d --ipc host --gpus all ghcr.io/cell-observatory/ops3d:develop_torch_26_01 bash


FROM nvcr.io/nvidia/pytorch:26.01-py3 AS base
ENV RUNNING_IN_DOCKER=TRUE

# Make bash colorful https://www.baeldung.com/linux/docker-container-colored-bash-output   https://ss64.com/nt/syntax-ansi.html 
ENV TERM=xterm-256color
RUN echo "PS1='\e[97m\u\e[0m@\e[94m\h\e[0m:\e[35m\w\e[0m# '" >> /root/.bashrc


# Install requirements. Don't "apt-get upgrade" or else all the NVIDIA tools and drivers will update.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  sudo \
  htop \
  cifs-utils \
  winbind \
  smbclient \
  sshfs \
  iputils-ping \
  google-perftools \
  libgoogle-perftools-dev \
  graphviz \
  zsh \
  vmtouch \
  fio \
  prometheus \ 
  autoconf \
  libxslt-dev \ 
  xsltproc \ 
  docbook-xsl \
  libnuma-dev \
  && rm -rf /var/lib/apt/lists/*


RUN echo "Install ohmyzsh"
RUN sh -c "$(wget https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O -)"

# Give the dockerfile the name of the current git branch (passed in as a command line argument to "docker build")
ARG BRANCH_NAME

# Want to rebuild from requirements.txt everytime, so if some new dependency breaks, we catch it right away.
# Therefore we must avoid cache in this next section https://docs.docker.com/reference/cli/docker/buildx/build/#no-cache-filter
# ----- Section to be non-cached when built.
FROM base AS pip_install
COPY requirements.txt requirements.txt 
# ------

FROM pip_install AS torch_26_01
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt --progress-bar off --root-user-action=ignore --cache-dir /root/.cache/pip 

RUN echo "Compile ops3d kernels"
# RUN python setup.py bdist_wheel
COPY dist/*.whl .

IF [ "$(uname -m)" = "x86_64" ]; then
    RUN pip install ops3d-0.1.0-cp312-cp312-linux_x86_64.whl
elif [ "$(uname -m)" = "aarch64" ]; then
    RUN pip install ops3d-0.1.0-cp312-cp312-linux_aarch64.whl
else
    RUN echo "Unsupported architecture"
    RUN exit 1
fi

# Code to avoid running as root
ARG USERNAME=user1000
ENV USER=${USERNAME}
ARG USER_UID=1000
ARG USER_GID=1000

# Create the user
RUN groupadd --gid $USER_GID $USERNAME && \
    groupadd --gid 1001 user1000_secondary && \
    useradd -l --uid $USER_UID --gid $USER_GID -G 1001 -m $USERNAME && \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.        
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME || true

# [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
