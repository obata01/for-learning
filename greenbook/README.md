# Docker
```
$ docker build --rm -t r-studio:latest .
$ docker run --rm -it -p 8787:8787 \
    -v $(pwd):/home/project \
    --name greenbook-container \
    r-studio:latest
```

```
docker run -d -p 8787:8787 \
    -v ~/for-learning/greenbook/.config/rstudio:/home/rstudio/.config/rstudio \
    -v $PWD:/home/rstudio/workspace \
    rocker/rstudio:4.0.3
```


