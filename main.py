import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
import requests


app = FastAPI()

class_labels = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'spider', 'sheep', 'squirrel']

# Load model architecture
model = models.densenet121(pretrained=False)

# Modify classifier to match Animals-10 (10 classes)
num_ftrs = model.classifier.in_features
model.classifier = torch.nn.Linear(num_ftrs, len(class_labels))

# Load trained weights
model.load_state_dict(torch.load('C:/Users/Chait/Desktop/API_Practise/densenet121_animals10.pth', map_location="cpu"))
model.eval()  # Set model to evaluation mode


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_image(image: Image.Image):
    """Preprocess the image and predict the class."""
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        predicted_class = output.argmax(1).item()
    return class_labels[predicted_class]

# Define a function to return a description of the app
def get_app_description():
	return (
    	"Welcome to the Animal Classification API!"
    	"This API allows you to predict the type of animals from a pool of 10 animals."
    	"Use the '/predict/' endpoint with a POST request to make predictions."
	)

# Define the root endpoint to return the app description
@app.get("/")
async def root():
	return {"message": get_app_description()}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """API endpoint to classify an uploaded image."""
    image = Image.open(BytesIO(await file.read()))  # Read image from request
    prediction = predict_image(image)
    return {"prediction": prediction}

@app.get("/predict_url/")
async def predict(image_url: str):
    """API endpoint to classify an image from a URL."""
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Check for request errors
        image = Image.open(BytesIO(response.content))
        prediction = predict_image(image)
        return {"prediction": prediction}
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
