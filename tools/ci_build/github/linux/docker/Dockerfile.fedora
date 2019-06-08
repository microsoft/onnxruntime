ARG OS_VERSION=27
FROM fedora:${OS_VERSION}
ADD scripts /tmp/scripts
RUN cd /tmp/scripts && /tmp/scripts/install_fedora.sh && /tmp/scripts/install_deps.sh && rm -rf /tmp/scripts

