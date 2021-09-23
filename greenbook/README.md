# Docker

下記を実行するとr-studio serverを起動することができる.

```
$ docker build --rm -t r-studio:latest .
$ docker run -d -p 8787:8787 \
      -v ~/for-learning/greenbook/.config/rstudio:/home/rstudio/.config/rstudio \
      -v $PWD:/home/rstudio/workspace \
      rocker/rstudio:4.0.3
```

localhost:8787 へアクセスし, デフォルトでは下記でログインするとr-studioを利用できる.

```
Username: rstudio
Password: rstudio
```



