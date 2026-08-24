import sys, os
sys.path.append(os.pardir)
import numpy as np
from PIL import Image
from model.deep_convnet import DeepConvNet
from common.functions import softmax
from preprocess import preprocess_image

PARAMS_PATH = "model/deep_convnet_params.pkl"

_network = None

def load_network(params_path=PARAMS_PATH):
    global _network
    if _network is None:
        _network = DeepConvNet()
        _network.load_params(params_path)

    return _network

def predict_digit(image):
    network = load_network()
    x = preprocess_image(image)

    scores = network.predict(x, train_flg=False)
    # probs.shape -> (1, 10)
    probs = softmax(scores)[0]

    predicted_digit = int(np.argmax(scores))
    confidence = float(probs[predicted_digit])

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "probabilities": probs.tolist(),
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = Image.open(image_path)
    result = predict_digit(img)

    print(f"Predicted digit : {result['predicted_digit']}")
    print(f"Confidence : {result['confidence']:.4f}")
    
