from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from RAG import RAGModel

app = FastAPI()

# Load the RAG model
rag_model = RAGModel()

# Note that we are using a very small data, mainly as a place holder as the actual data requires pre-processing to be useful. 
with open('data/sample_data.txt', 'r') as f:
    data = f.read().splitlines()
rag_model.index_data(data)

# classification model
with open('iris_classifier.pkl', 'rb') as file:
    clf = pickle.load(file)

class Query(BaseModel):
    text: str

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
async def root():
    return {"message": "Welcome to the Infiheal assignment project"}
    
@app.post("/rag")
async def generate(query: Query):
    result = rag_model.generate(query.text)
    return {"generated_text": result}

@app.post("/classification")
async def classify(features: IrisFeatures):
    # Convert input to numpy array
    input_data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])
    
    # Making prediction
    prediction = clf.predict(input_data)
    
    # Deciding the class name
    iris_classes = ['setosa', 'versicolor', 'virginica']
    predicted_class = iris_classes[prediction[0]]
    
    return {"predicted_class": predicted_class}