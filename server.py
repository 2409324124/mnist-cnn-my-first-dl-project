"""
server.py — Flask API 服务器
职责: HTTP 路由、接收前端图片、调用 inference 模块、返回 JSON
不涉及任何模型/ML 逻辑
"""

import base64
import io
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import inference   # 唯一与 ML 的耦合点

app = Flask(__name__, static_folder="static")


# ── 路由：前端页面 ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ── 路由：推理 API ────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "missing 'image' field"}), 400

    # 解码 base64 PNG（来自 canvas.toDataURL）
    try:
        header, encoded = data["image"].split(",", 1)
        img_bytes = base64.b64decode(encoded)
        pil_image = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        return jsonify({"error": f"image decode failed: {e}"}), 400

    probs = inference.predict(pil_image)

    return jsonify({
        "prediction": int(probs.index(max(probs))),
        "probabilities": {str(i): round(probs[i], 4) for i in range(10)}
    })


# ── 启动 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("启动 MNIST 识别服务 → http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
