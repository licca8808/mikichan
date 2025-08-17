# app.py

# 必要なライブラリをインポートします。
import torch # type: ignore
import requests # type: ignore # HTTPリクエストを送信するためのライブラリ
import json # JSONデータの処理に必要
# ここが最も重要です！ Flaskから session, redirect, url_for を正しくインポートしているか確認してください
from flask import Flask, request, render_template, redirect, url_for, session # type: ignore
from sentence_transformers import SentenceTransformer, util # type: ignore

# Flaskアプリのインスタンスを作成します。
app = Flask(__name__)
# Flaskのセッション機能を使うために秘密鍵を設定します。
# ここを「'your_secret_key_for_session_management'」から、任意のランダムな文字列に置き換える！重要！
app.secret_key = 'licca8808'

# --- AIモデルのロード ---
print("Loading SentenceTransformer model...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("SentenceTransformer model loaded.")

# 感情分析用のGemini API呼び出し関数
def analyze_sentiment_with_gemini(text_to_analyze):
    # Canvas環境ではAPIキーは空文字列でOKです。ご自身でキーをここに直接記述しないでください。
    api_key = "" # Canvas環境では空文字列でOKです。ご自身でキーをここに直接記述しないでください。
    model_name = "gemini-2.5-flash-preview-05-20"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    prompt = f"""
    以下のテキストの感情を「POSITIVE」「NEGATIVE」「NEUTRAL」のいずれかで分類し、
    その分類結果と確信度をJSON形式で出力してください。出力はJSONオブジェクトのみとし、
    それ以外のテキストは含めないでください。

    例:
    テキスト: 「このレストランはとても美味しくて最高でした！」
    出力: {{"label": "POSITIVE", "score": 0.95}}

    テキスト: 「騒がしくて、料理もいまいちだった。」
    出力: {{"label": "NEGATIVE", "score": 0.88}}

    テキスト: 「特に何も感じなかった。」
    出力: {{"label": "NEUTRAL", "score": 0.60}}

    テキスト: 「{text_to_analyze}」
    出力:
    """

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
        }
    }

    try:
        response = requests.post(api_url, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()

        result = response.json()
        
        if result.get("candidates") and result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts"):
            text_response = result["candidates"][0]["content"]["parts"][0]["text"]
            parsed_json = json.loads(text_response)
            return parsed_json
        else:
            print("Gemini APIからの応答構造が予期せぬものでした:", result)
            return {"label": "UNKNOWN", "score": 0.0}
    except requests.exceptions.RequestException as e:
        print(f"Gemini API呼び出しエラー: {e}")
        return {"label": "ERROR", "score": 0.0, "message": str(e)}
    except json.JSONDecodeError as e:
        print(f"Gemini API応答のJSONパースエラー: {e}")
        print(f"応答テキスト: {text_response if 'text_response' in locals() else 'N/A'}")
        return {"label": "ERROR", "score": 0.0, "message": "JSONパースエラー"}


# --- ダミーデータ（お店の情報）---
restaurants_data = [
    {
        "name": "喫茶店 木漏れ日",
        "atmosphere_description": "静かで落ち着いた雰囲気で、読書や一人でくつろぐのに最適です。穏やかなBGMが流れています。",
        "exterior_image_url": "https://placehold.co/400x250/E6B08F/FFFFFF?text=Cafe+Exterior",
        "menu_text": "本日のコーヒー、手作りケーキ、サンドイッチなど",
        "product_image_url": "https://placehold.co/300x200/F5CBA7/FFFFFF?text=Coffee+%26+Cake",
        "average_price": 800,
        "address": "東京都渋谷区",
        "customer_demographics_score": 3,
    },
    {
        "name": "活気ある居酒屋 🏮笑顔亭",
        "atmosphere_description": "ワイワイ賑やかで、友人と楽しくお酒を飲むのにぴったりです。活気があります。",
        "exterior_image_url": "https://placehold.co/400x250/C8A2C8/FFFFFF?text=Izakaya+Exterior",
        "menu_text": "焼き鳥、刺身、ビール、日本酒",
        "product_image_url": "https://placehold.co/300x200/FFD700/FFFFFF?text=Yakitori+%26+Beer",
        "average_price": 3500,
        "address": "大阪府中央区",
        "customer_demographics_score": 8,
    },
    {
        "name": "オーガニックレストラン 緑の風",
        "atmosphere_description": "自然光が差し込む明るく開放的な空間で、健康的な食事が楽しめます。ベジタリアンフレンドリー。",
        "exterior_image_url": "https://placehold.co/400x250/8FBC8F/FFFFFF?text=Organic+Restaurant",
        "menu_text": "季節の野菜プレート、スムージー、グルテンフリーパン",
        "product_image_url": "https://placehold.co/300x200/A2D9CE/FFFFFF?text=Veggie+Plate",
        "average_price": 2000,
        "address": "福岡県天神",
        "customer_demographics_score": 5,
    },
    {
        "name": "隠れ家バー Moonlight",
        "atmosphere_description": "落ち着いた照明で、大人の雰囲気が漂うシックなバーです。デートやゆっくり過ごしたい時に。",
        "exterior_image_url": "https://placehold.co/400x250/465B7C/FFFFFF?text=Bar+Exterior",
        "menu_text": "カクテル、ウイスキー、軽食",
        "product_image_url": "https://placehold.co/300x200/708090/FFFFFF?text=Cocktail",
        "average_price": 5000,
        "address": "北海道札幌市",
        "customer_demographics_score": 4,
    },
    {
        "name": "家族で楽しめるレストラン ハッピーダイニング",
        "atmosphere_description": "子供連れに優しい、明るくカジュアルな雰囲気のレストランです。広々とした空間で食事を楽しめます。",
        "exterior_image_url": "https://placehold.co/400x250/ADD8E6/FFFFFF?text=Family+Restaurant",
        "menu_text": "お子様ランチ、ハンバーグ、パスタ",
        "product_image_url": "https://placehold.co/300x200/FFDAB9/FFFFFF?text=Kids+Meal",
        "average_price": 1500,
        "address": "沖縄県那覇市",
        "customer_demographics_score": 7,
    },
    {
        "name": "モダンな和食処 結",
        "atmosphere_description": "洗練されたモダンな空間で、本格的な和食を静かに楽しめます。接待や特別な日にも最適です。",
        "exterior_image_url": "https://placehold.co/400x250/808080/FFFFFF?text=Japanese+Restaurant",
        "menu_text": "寿司、天ぷら、会席料理",
        "product_image_url": "https://placehold.co/300x200/D3D3D3/FFFFFF?text=Sushi",
        "average_price": 6000,
        "address": "京都府京都市",
        "customer_demographics_score": 2,
    },
    {
        "name": "ブックカフェ 栞",
        "atmosphere_description": "多くの本が並ぶ静かな空間で、コーヒーを飲みながら読書を楽しめます。作業にも集中できます。",
        "exterior_image_url": "https://placehold.co/400x250/A9A9A9/FFFFFF?text=Book+Cafe",
        "menu_text": "スペシャルティコーヒー、軽食、デザート",
        "product_image_url": "https://placehold.co/300x200/C0C0C0/FFFFFF?text=Book+Cafe+Interior",
        "average_price": 900,
        "address": "神奈川県横浜市",
        "customer_demographics_score": 3,
    }
]

restaurant_embeddings = model.encode([r["atmosphere_description"] for r in restaurants_data], convert_to_tensor=True)


# --- ルート定義 ---

# アプリのランディングページ (index.html をレンダリング)
# 例: http://127.0.0.1:5000/ にアクセスしたときに表示される
@app.route('/', methods=['GET'])
def landing_page():
    return render_template('index.html') # index.html をレンダリング

# 検索フォームページ (recomend.html をレンダリング)
# 例: http://127.0.0.1:5000/search にアクセスしたときに表示される
@app.route('/search', methods=['GET'])
def search_form():
    # セッションからエラーメッセージを取得し、テンプレートに渡す
    # error_message は、もしセッションになければ None になります。
    error_message = session.pop('error_message', None)
    return render_template('recomend.html', error_message=error_message)  # recomend.html をレンダリング

# 検索結果ページ（フォームからのPOSTを受け取り、result.html をレンダリング）
# 例: フォームが /recommend にデータを送信するとこのルートが処理する
@app.route('/recommend', methods=['POST'])
def recommend_restaurant():
    user_atmosphere = request.form.get('atmosphere', '')
    recommended_restaurant = None
    top_restaurants = []
    user_sentiment = None
    # ここに error_message の初期化を追加します
    error_message = None

    if not user_atmosphere:
        # 入力がない場合はエラーメッセージをセッションに保存して検索フォームページにリダイレクト
        session['error_message'] = "雰囲気を入力してください！"
        return redirect(url_for('search_form')) # search_form ルートにリダイレクト
    else:
        try:
            # --- 感情分析機能（Gemini APIを使用）---
            user_sentiment = analyze_sentiment_with_gemini(user_atmosphere)
            print(f"ユーザー入力の感情分析結果 (Gemini): {user_sentiment}")

            # --- 雰囲気マッチング ---
            user_embedding = model.encode(user_atmosphere, convert_to_tensor=True)
            similarities = util.cos_sim(user_embedding, restaurant_embeddings)[0]

            best_match_index = torch.argmax(similarities).item()
            recommended_restaurant = restaurants_data[best_match_index]
            recommended_restaurant["similarity_score"] = round(similarities[best_match_index].item(), 4)

            sorted_indices = torch.argsort(similarities, descending=True)
            top_n = 3
            
            count = 0
            for i in range(len(restaurants_data)):
                idx = sorted_indices[i].item()
                if idx == best_match_index:
                    continue
                if count < top_n -1:
                    restaurant = restaurants_data[idx].copy()
                    restaurant["similarity_score"] = round(similarities[idx].item(), 4)
                    top_restaurants.append(restaurant)
                    count += 1
                else:
                    break

        except Exception as e:
            # エラーが発生した場合もエラーメッセージをセッションに保存して検索フォームページにリダイレクト
            error_message_text = f"お店の検索中にエラーが発生しました: {e}"
            print(error_message_text)
            session['error_message'] = error_message_text
            return redirect(url_for('search_form')) # search_form ルートにリダイレクト

    # 全てが成功した場合、結果ページをレンダリングしてデータを渡す
    # ここに error_message=error_message を追加します
    return render_template(
        'result.html', # result.html をレンダリング
        recommended_restaurant=recommended_restaurant,
        top_restaurants=top_restaurants,
        user_sentiment=user_sentiment,
        error_message=error_message # error_message をテンプレートに渡す
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
