from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import chatbot_response

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    msg = data.get("message", "")
    response = chatbot_response(msg)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
