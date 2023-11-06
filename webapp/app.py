## app.py
# ONNX 런타임 세션을 생성하기 위해 필요한 라이브러리들을 임포트
from flask import Flask, request, jsonify
import torch
import numpy as np
from transformers import RobertaTokenizer
import onnxruntime


# 플라스크 애플리케이션 생성
app = Flask(__name__)

# 토큰화 모듈 정의
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# 모델의 경로와 함께 ONNX 런타임을 초기화
session = onnxruntime.InferenceSession(
		"/ws/roberta-sequence-classification-9.onnx")

# /predict 경로로 들어오는 POST 요청을 처리하는 라우팅 함수
@app.route("/predict", methods=["POST"])
def predict():
		input_ids = torch.tensor(
				tokenizer.encode(request.json[0], add_special_tokens=True)
		).unsqueeze(0)
		if input_ids.requires_grad:
				x = input_ids.detach().cpu().numpy()
		else:
				x = input_ids.cpu().numpy()
		inputs = {session.get_inputs()[0].name: x}
		out = session.run(None, inputs)
		result = np.argmax(out)
		return jsonify({"positive": bool(result)})

if __name__=="__main__":
		app.run(host="0.0.0.0", port=5001, debug=True)
