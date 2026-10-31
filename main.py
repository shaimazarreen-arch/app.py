import os
import cv2
import numpy as np
import mediapipe as mp
import base64
from flask import Flask, request, jsonify

app = Flask(__name__)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

@app.route('/analyze-symmetry', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "Missing image data"}), 400

    try:
        img_data = base64.b64decode(data['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        h, w, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return jsonify({"error": "No face detected"}), 400
            
        landmarks = results.multi_face_landmarks.landmark
        coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
        
        # Select nose, left mouth, right mouth coordinates for symmetry index mapping
        nose_tip = coords[4]
        left_mouth = coords[61]
        right_mouth = coords[291]
        
        mouth_left_dist = np.linalg.norm(nose_tip - left_mouth)
        mouth_right_dist = np.linalg.norm(nose_tip - right_mouth)
        mouth_asymmetry = abs(mouth_left_dist - mouth_right_dist) / max(mouth_left_dist, mouth_right_dist, 1)
        
        return jsonify({
            "status": "success",
            "asymmetry_score": float(mouth_asymmetry)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
