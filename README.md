# StanとRでベイズ統計モデリング（アヒル本）

## 環境構築

### クローン＆コンテナ起動
```
$ git clone -b duck-book git@github.com:obata01/for-learning.git duck-book
$ cd duck-book
$ docker-compose up
```

### VSCodeからコンテナへアタッチ

- Remote - Containers  の拡張機能をインストール<br>
  → duck-bookコンテナへアタッチ

### VSCodeへR拡張機能をインストール
下記拡張機能をインストール

1. R
2. R Tools

### Rstanのインストール
```
$ R
> install.packages('rstan', dependencies=TRUE)
```

