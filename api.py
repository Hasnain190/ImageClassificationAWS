from flask import Flask, request, jsonify
app = Flask(__name__)
from application import predict , TOP_CLASSES 

import base64

@app.route('/predict', methods=['POST'])
def predict_api():
    # image = request.json
    
    if request.content_type == 'application/json':
        try:
            data = request.get_json()
            image_data = data.get('image_data')
            decoded_image = base64.b64decode(image_data)
            prediction = predict(decoded_image)
            top_4_classes = prediction.argsort()[0][-TOP_CLASSES:][::-1]

            # Convert the NumPy array to a Python list
            top_4_classes_list = top_4_classes.tolist()

            return jsonify(top_4_classes_list)
        except Exception as e:
            return jsonify({'error': str(e)})
        
    else:
        return jsonify({'error': 'Unsupported Media Type'})
    
   
    
    
    
    # image = './00000061_013.png'
    # prediction = predict(image)
    # top_4_classes = prediction.argsort()[0][-TOP_CLASSES:][::-1]

    # # Convert the NumPy array to a Python list
    # top_4_classes_list = top_4_classes.tolist()

    # return jsonify(top_4_classes_list)
    
   

if __name__ == '__main__':
    app.run()
