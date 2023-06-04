# AWSでの環境構築手順

1. AMIからインスタンスを起動
    - リージョンを「バージニア北部」等へ変更
    - AMIカタログ画面で「GPU」で検索して任意のAMIを選択
    - 「AMIでインスタンスを起動」を押下
    - パブリックIPを許可
    - us-east1-[a~d]のVPCサブネットを利用
    - ストレージを300G選択
    - 作成する

2. SSH接続

EC2インスタンスの画面の右上の「接続＞SSHクライアント」からSSHコマンド取得しログイン

3. Docker install

以下でDockerをインストールする。

- [Dockerのリポジトリを設定する](https://sid-fm.com/support/vm/guide/install-docker-ubuntu.html#repository)
- [Docker をインストールする](https://sid-fm.com/support/vm/guide/install-docker-ubuntu.html#install)

4. GitHub設定

- ssh-keyの生成
```
ssh-keygen -t rsa
```

- GitHubへ公開鍵を登録
下記公開鍵をGitHubの自分のアカウントへ設定する
```
cat ~/.ssh/id_rsa.pub
```

5. 環境構築

```
docker-compose up -d
```

6. jupyter起動
```
jupyter lab --allow_root --ip="0.0.0.0"
```

http://[ec2のホスト]:8888  へアクセス  
例）http://ec2-35-153-133-218.compute-1.amazonaws.com:8888/lab/tree/notebooks  
※アクセスできない場合はセキュリティーグループを確認


7. 必要なデータのダウンロード

```
wget -P ${APP_DIR}/data -i luna_file_path.txt
```
※環境変数APP_DIRはDockerfile内で定義している。


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

