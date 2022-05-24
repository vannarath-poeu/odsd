# ODSD

On-Device Spam Detection

## Folder structure

This monorepo contains all parts needed to develop ODSD. 
- APP: PWA based on React and Material UI to demonstrate a messaging application.
- Jupyter: Exploratory work to develop models.

(Transient folders)
- Data: central place to keep data gathered.

## How to:
- Assumes docker is installed.
- Assumes knowledge of Makefile. If your system does not support Makefile, replace make command with the correspending commands.
- run `make scraper-up` to download data.
- run `make jupyterlab-up` to start notebook (Copy link printed in terminal).
- run `make app-up` to start React app. This will be used as the front-end for user interactions. Note: the first run is extremely slow while packages are being installed.

(Optional): the above commands have a down version where you replace the `-up` with `-down` to destroy the dockers. This should be used when drastic changes are made and cache needs to be destroyed.