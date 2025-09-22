from flask import Flask, request, jsonify
from auth import generate_token
from users_db import users

app = Flask(__name__)

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    user = users.get(username)

    if not user:
        return jsonify({"error": "User not found"}), 404

    from passlib.hash import bcrypt
    if not bcrypt.verify(password, user["password"]):
        return jsonify({"error": "Invalid password"}), 401

    token = generate_token(username, user["role"])
    return jsonify({"token": token})


# Örnek korumalı endpoint: Sadece admin ve analyst erişebilir
from auth import login_required

@app.route("/run-detection", methods=["POST"])
@login_required(roles=["admin", "analyst"])
def run_detection():
    return jsonify({"message": f"Anomaly detection started by {request.user['username']}"})

if __name__ == "__main__":
    app.run(debug=True)
