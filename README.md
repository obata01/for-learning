# GCP VertexAI WorkBenchでの学習環境構築


## GCPアカウントの用意
1. gmailアカウントを新規作成

e.g.) gcp01.xxxxxx@gmail.com

2. GCPへログイン
https://console.cloud.google.com/welcome

3. 新規project作成

4. 無料トライアルを始める

5. vertex ai workbench を有効にして利用を始める
https://cloud.google.com/vertex-ai-workbench?hl=ja

## GPUを利用できるようにする（有料アカウントへ移行する）
GCPの無料アカウントではGPUを利用できないため有料アカウントに移行する必要がある。
https://cloud.google.com/free/docs/free-cloud-features?hl=ja#how-to-upgrade


## GCPからGitHubを操作する
詳細は下記を参照
https://docs.github.com/ja/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent

GitHub側の仕様変更等がない限り基本的には下記でOK

1. ssh-keyの生成
```
ssh-keygen -t rsa
```

2. GitHub
下記公開鍵をGitHubの自分のアカウントへ設定する
```
cat ~/.ssh/id_rsa.pub
```


## VertexAI WorkBenchを始める

1. cloud storageのバケットを作成する

2. 新しいノートブックを作成する(マネージドノートブックの方がよい？)

3. JupyterLabを開く

4. git clone

```
git clone -b pytorch git@github.com:obata01/for-learning.git luna
cd luna
```

5. ライブラリインストール

```
pip install -r requirement.txt
```

6. 必要なデータのダウンロード

```
export LUNA_HOME="/home/jupyter/luna"
wget -P ${LUNA_HOME}/data -i luna_file_path.txt
```

7. 必要に応じてgit push

## GPUを利用する
GCPの無料アカウントではGPUを利用できないため有料アカウントに移行する必要がある。

https://cloud.google.com/free/docs/free-cloud-features?hl=ja#how-to-upgrade

