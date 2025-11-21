from flask import Flask, request, jsonify
from chatbot import get_response   # <-- the function that generates chatbot replies

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "No 'message' field found."}), 400

    user_msg = data["message"]
    bot_response = get_response(user_msg)

    return jsonify({"response": bot_response})
    
if __name__ == "__main__":
    app.run(debug=True)
