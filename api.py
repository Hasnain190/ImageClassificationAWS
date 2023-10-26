from flask import Flask, request, jsonify
app = Flask(__name__)
from application import predict , TOP_CLASSES 
from PIL import Image
import io
import base64

@app.route('/predict', methods=['POST'])
def predict_api():
    # image = request.json
    
    if request.content_type == 'application/json':
        try:
            data = request.get_json()
            image_data = data.get('image_data')
            
           
            img = convert_to_img(image_data)
           
            prediction = predict(img)
            top_4_classes = prediction.argsort()[0][-TOP_CLASSES:][::-1]

            # Convert the NumPy array to a Python list
            top_4_classes_list = top_4_classes.tolist()

            return jsonify(top_4_classes_list)
        except Exception as e:
            return jsonify({'error': str(e)})
        
    else:
        return jsonify({'error': 'Unsupported Media Type'})


def convert_to_img(image_data):
    decoded_image = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(decoded_image))
    return image


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"
    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
