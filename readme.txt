ディレクトリ
your-app-name/
├── app.py
├── templates/
│   └── index.html
└── static/
    └── style.css

必要なライブラリのインストール
# 仮想環境を作成 (初回のみ)
conda create -n restaurant_env python=3.9

# 仮想環境をアクティベート
# Windows
conda activate restaurant_env
# macOS/Linux
conda activate restaurant_env

# 必要なライブラリをインストール
pip install Flask 
pip install sentence-transformers 
pip install transformers 
pip install torch
pip install numpy==1.25.2

一度作ったらactivateするだけ