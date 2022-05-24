jupyterlab-up:
	docker-compose -f jupyterlab/docker/docker-compose.yaml up

jupyterlab-down:
	docker-compose -f jupyterlab/docker/docker-compose.yaml down --volumes --rmi all

app-up:
	docker-compose -f app/docker/docker-compose.yaml up

app-down:
	docker-compose -f app/docker/docker-compose.yaml down --volumes --rmi all