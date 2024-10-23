FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git \
    && curl -fsSL https://code-server.dev/install.sh | bash  -s -- --version 4.93.1 --prefix=/usr/local --method=standalone \
    && curl -ssL https://magic.modular.com/2e832d37-0df5-4261-aa3d-d324e25c863d | bash \
    && apt-get clean \
    && rm -rf /config/* /tmp/* /var/lib/apt/lists/* /var/tmp/*

CMD ["/usr/local/bin/code-server", "--bind-addr", "0.0.0.0:8080", "--auth", "none"]