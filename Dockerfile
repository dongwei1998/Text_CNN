FROM



# time zone set
WORKDIR /usr/share
ADD ./zoneinfo ./zoneinfo
RUN  ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
RUN echo "Asia/Shanghai" > /etc/timezone