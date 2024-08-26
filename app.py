from flask import Flask, request, jsonify
from utils1 import find_paper, read_id, get_ans_200,get_ans_100
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/data', methods=['POST'])
def data():
    # Retrieve the uploaded file and sheet_type from the request
    file = request.files['image']
    sheet_type = request.form.get('sheet_type')
    
    if not file:
        return jsonify({"error": "No image uploaded"}), 400
    if sheet_type not in ['100', '200']:
        return jsonify({"error": "Invalid sheet type provided. Must be '100' or '200'."}), 400

    # Convert the uploaded image to OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    org_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Find the OMR sheet in the image
    OMR = find_paper(org_image)

    # Process based on the selected sheet_type
    final_ans = []
    
    if sheet_type == '200':
        # Extract student ID and test ID for both 100 and 200-question sheets
        s_id = OMR[55:245, 10:260]
        t_id = OMR[55:250, 281:365]
        
        # Read the student ID and test ID
        student_id = read_id(s_id, 10)
        test_id = read_id(t_id, 3)
        # Process 200-question sheet
        q1 = OMR[300:1287, 54:184]
        q2 = OMR[300:1287, 256:386]
        q3 = OMR[300:1287, 456:586]
        q4 = OMR[300:1287, 658:788]
        q5 = OMR[300:1287, 859:989]
        
        # Read answers from all sections
        q1_ans = get_ans_200(q1)
        q2_ans = get_ans_200(q2)
        q3_ans = get_ans_200(q3)
        q4_ans = get_ans_200(q4)
        q5_ans = get_ans_200(q5)
        
        # Combine answers into final list
        final_ans = q1_ans + q2_ans + q3_ans + q4_ans + q5_ans

    elif sheet_type == '100':    
        # student id and test ID image
        s_id=OMR[60:278,9:257]
        t_id=OMR[60:278,281:365]
        
        # Read the student ID and test ID
        student_id = read_id(s_id, 10)
        test_id=read_id(t_id,3)

        # Process 100-question sheet
        q1 = OMR[338:1287, 50:184]
        q2 = OMR[338:1287, 252:386]
        q3 = OMR[338:1287, 455:589]
        q4 = OMR[338:1287, 658:792]
        q5 = OMR[338:1287, 859:993]
        
        # Read answers from all sections
        q1_ans = get_ans_100(q1)
        q2_ans = get_ans_100(q2)
        q3_ans = get_ans_100(q3)
        q4_ans = get_ans_100(q4)
        q5_ans = get_ans_100(q5)
        
        # Combine answers into final list
        final_ans = q1_ans + q2_ans + q3_ans + q4_ans + q5_ans

    # Convert answers to strings
    final_ans = list(map(str, final_ans))

    # Return the response as JSON
    return jsonify({"student_id": student_id, "test_id": test_id, "Ans": final_ans})

if __name__ == '__main__':
    app.run(debug=True)
