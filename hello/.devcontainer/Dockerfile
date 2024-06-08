FROM ubuntu:23.04
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install python3-pip python3-dev curl git \
    && rm -f /usr/lib/python3.11/EXTERNALLY-MANAGED \
    && curl https://get.modular.com | MODULAR_AUTH=mut_6822a0a033564641beeacea073336120 sh - \
    && modular install mojo \
    && echo 'export MODULAR_HOME="$HOME/.modular"' >> ~/.bashrc \
    && echo 'export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"' >> ~/.bashrc \
    && source ~/.bashrc