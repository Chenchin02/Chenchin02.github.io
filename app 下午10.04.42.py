
from flask import Flask, request, jsonify, g, render_template
from chatPDF import DocumentChatAssistant
import threading
import pandas as pd
import os


def save_conversation_to_excel(user_message, bot_response):
    file_path = '/Users/chenchin/Downloads/flaskProject/static/conversation_history.xlsx'
    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
    else:
        df = pd.DataFrame(columns=['使用者問題', '機器人回應'])

    # 使用 concat 而不是 append
    new_record = pd.DataFrame([[user_message, bot_response]], columns=['使用者問題', '機器人回應'])
    df = pd.concat([df, new_record], ignore_index=True)

    # 保存 DataFrame 到 Excel
    df.to_excel(file_path, index=False)


app = Flask(__name__)
assistant = DocumentChatAssistant(openai_api_key="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")

@app.before_request
def before_request():
    # 每次請求前初始化 history
    g.history = []

@app.teardown_request
def teardown_request(exception):
    # 每次請求結束後，清掉 history
    g.history = []


@app.route("/")
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        data = request.json
        inp = data.get('message')

        response = assistant.main(inp, g.history)

        # 異步保存對話記錄
        save_thread = threading.Thread(target=save_conversation_to_excel, args=(inp, response[0][-1]))
        save_thread.start()

        return jsonify({'status': 'success', 'response_message': response[0][-1]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
