import jwt
from flask import request, jsonify
from functools import wraps
from users_db import users, SECRET_KEY

def generate_token(username, role):
    return jwt.encode({"username": username, "role": role}, SECRET_KEY, algorithm="HS256")

def decode_token(token):
    return jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

def login_required(roles=None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            token = request.headers.get("Authorization")
            if not token or not token.startswith("Bearer "):
                return jsonify({"error": "Token missing"}), 401
            try:
                payload = decode_token(token.split()[1])
                if roles and payload["role"] not in roles:
                    return jsonify({"error": "Forbidden"}), 403
                request.user = payload  # kullanıcının bilgisi route içinde erişilebilir
            except Exception:
                return jsonify({"error": "Invalid token"}), 401
            return f(*args, **kwargs)
        return wrapper
    return decorator
