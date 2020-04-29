---
title: 'A short Docker tutorial'
date: 2020-01-16
permalink: /posts/2020/01/short-docker-tutorial/
tags:
  - tools
  - cloud
---

### Dockerfile ###
here is a example Dockerfile:
```
FROM node:alpine

WORKDIR '/app'

COPY package.json .
RUN npm install

COPY . .
CMD ["npm", "run", "start"]
```
below we will explain the usage of these commands:
```
FROM node:alpine
```
it just search for the base image hub and pulls it if necessary
```
WORKDIR '/app'
```
set the /app folder inside the docker as the working directory
```
COPY package.json .
```
copy packages.json under current local working directory to docker working directory
why we do this? because this copy is very time-consuming, just install it insider docker is much faster!
```
RUN npm install
```
install...

```
COPY . .
```
copy other src code file to docker working directory
```
CMD ["npm", "run", "start"]
```
docker starting command 

### Dockercompose file ###
```
version: '3'
services:
  web: 
    build: 
      context: .
      dockerfile: Dockerfile.dev
    ports: 
      - "3000:3000"
    volumes:
      - /app/node_modules
      - .:/app
```
docker compose provides more complex functionality to manage multiples containers, like set up tcp connections between two dockers(e.g backend service docker and front end docekr)
```
version: '3'
```

specifies the version of docker-compose? 
```
services:
```
then you name and specify configs of all the containers under services option.
```
web: 
    build: 
      context: .
      dockerfile: Dockerfile.dev
    ports: 
      - "3000:3000"
    volumes:
      - /app/node_modules
      - .:/app
```
first we have  the web service container built from Dockerfile.dev under current directory,
```
 context: .
 ```
 specifies where to find the Dockerfile.dev, ports still works mapping ports of docker and localhost.

```
volumes:
    - /app/node_modules
    - .:/app 
```
bookmark volumes, 
```
 - /app/node_modules
```
this one has no ':' mappingg mark and it just means use the  /app/node_modules inside the docker but not  /app/node_modules folder in the local current directory, other than that directory we like previouly map them to /app directory of the docker, which we have specifiedg as the working directory in previous Dockerfile.

### Override starting command  ###
after you build your image, you can run the container overriding the starting command specified in the original docker file like
```
docker run wangdong/frontend npm run test
```
use '-it' flags to specify the interactive input when entering docker.
```
docker run -it wangdong/frontend npm run test
```
### Live test ###
```
docker-compose up
```
Builds, (re)creates, starts, and attaches to containers for a service. When run another terminal window with "docker exec -it CONTAINER_ID npm run test" and you changes the local test file the tests runner will be updated lively.

### Another Solution to run test suite lively ###
the above solution needs to first build up the container and then exec the commands inside that container which is not convinient. Actually we can create another service to run the tests tasks alone. 
```
%%%web service%%%
tests:
  build:
    context: .
    dockerfile: Dockcerfile.dev
  volumes:
    - /app/node_modules
    - .:/app
```

Now the Docker-compose file becomes:
```
version: '3'
services:
  web: 
    build: 
      context: .
      dockerfile: Dockerfile.dev
    ports: 
      - "3000:3000"
    volumes:
      - /app/node_modules
      - .:/app
  tests:
    build:
      context: .
      dockerfile: Dockcerfile.dev
    volumes:
      - /app/node_modules
      - .:/app
    command: ["npm", "run", "test"]
```

then when executing "docker-compose up --build" then there are two containers built up, one works as the web server and one for running test.

### support live input ###
Although above works, but we cannot directly input from the terminal to pipe into the tests docker stdin. If we want to do that, we can find the container image and exec
```
docker exec -it CONTAINER id sh
```

