# builder for installing python packages
FROM python:3.10.14-bookworm as builder

# install mlflow and psycopg2
RUN pip install \
    mlflow \
    psycopg2 \
    pysftp 

# main image
FROM python:3.10.14-slim-bookworm

# copy python packages
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
# copy mlflow package
COPY --from=builder /usr/local/bin/mlflow /usr/local/bin/mlflow
# copy linux packages for Postgres
COPY --from=builder /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
COPY --from=builder /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
# copy gunicorn package
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin/gunicorn

# open port 5000
EXPOSE 5000

# set environment variables
ARG POSTGRES_USER
ARG POSTGRES_PASSWORD
ARG DB_HOST
ARG DB_NAME
ARG ARTIFACTS_DESTINATION  
ENV POSTGRES_USER=${POSTGRES_USER}
ENV POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
ENV DB_HOST=${DB_HOST}
ENV DB_NAME=${DB_NAME}
ENV ARTIFACTS_DESTINATION=${ARTIFACTS_DESTINATION}

# Start mlflow server
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${DB_HOST}:5432/${DB_NAME} \
    --artifacts-destination ${ARTIFACTS_DESTINATION} 
