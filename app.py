import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        print(x.shape)
        x = x.unsqueeze(0)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

app = Flask(__name__)
CORS(app)

state_dict = torch.load("my_model.pth")
model = NeuralNetwork()
model.load_state_dict(state_dict)

model.eval()
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = np.array(int_features, dtype=np.float32)
    input_tensor = torch.tensor(final_features, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = torch.argmax(outputs.data).item()
        probability = torch.softmax(outputs.data, dim=1)[0][predicted_class].item() 
      
    return render_template('index.html', prediction_text='Output from Neural Network $ {}'.format(predicted_class))

if __name__ == "__main__":
    app.run(port=8000,debug=True)