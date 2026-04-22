from flask import Flask, jsonify
import json

app = Flask(__name__)

@app.route("/shots")
def get_shots():
    with open("shot_events.json", "r") as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
