# MLflow Tracking Environment Setup Guide

## Setup Servers
### Storage Server
In this setup, we will use Illinois Campus Cluster (ICC) to store the artifacts.

1. Add SSH public key to the campus cluster from the machine where you run the MLflow tracking server.
2. Test to connect to the campus cluster without your password.
   1. If unsuccessful, make sure:
      1. your public key is correctly registered in `~/.ssh/authorized_keys` on your ICC workspace;
      2. your login information is listed in `~/.ssh/config` on your local workspace.
3. Make sure `pysftp` is installed via pip/poetry in the server and client.
4. Test to connect to ICC's data transfer node (DTN) by `ssh user@cc-xfer.campuscluster.illinois.edu`
5. Run MLflow server on CLI by running 
   1. ```mlflow server --host 0.0.0.0 --port 8885 --artifacts-destination sftp://user@cc-xfer.campuscluster.illinois.edu/your/path/to/dir```
6. Run `scripts/sklearn_example.py` to see if it correctly runs.