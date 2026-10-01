from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from embedding_generator import EmbeddingGenerator
import torch
import torch.nn as nn
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir la arquitectura del modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1280, hidden_size=256, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Tomamos solo la última salida del LSTM
        last_output = lstm_out[:, -1, :]
        # Aplicamos la capa fully connected
        output = self.fc(last_output)
        return output

app = FastAPI()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))

# Verificar token de Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("HF_TOKEN no encontrado en variables de entorno. Algunas funcionalidades pueden no estar disponibles.")

try:
    # Cargar modelo PyTorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando dispositivo: {device}")
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "lstm_best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en {model_path}")
    
    # Crear instancia del modelo
    model = LSTMModel().to(device)
    
    # Cargar el estado del modelo
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)
    else:
        model = state_dict.to(device)
    
    model.eval()  # Set the model to evaluation mode
    logger.info("Modelo LSTM cargado exitosamente")

    # Inicializar el generador de embeddings
    generator = EmbeddingGenerator(token=HF_TOKEN)
    logger.info("Generador de embeddings inicializado")
except Exception as e:
    logger.error(f"Error durante la inicialización: {str(e)}")
    raise

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def form_post(request: Request, sequence: str = Form(...)):
    try:
        if not sequence:
            raise HTTPException(status_code=400, detail="La secuencia no puede estar vacía")
            
        # Generar embeddings
        embedding, _ = generator.generate_embeddings([sequence], ["input"])
        
        # Convertir a tensor de PyTorch y ajustar la forma
        input_tensor = torch.FloatTensor(np.array(embedding)).to(device)
        # Asegurar que el tensor tenga la forma correcta (batch_size, seq_len, input_size)
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Realizar predicción
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.cpu().numpy()[0]
        
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": prediction, 
            "sequence": sequence
        })
    except Exception as e:
        logger.error(f"Error durante la predicción: {str(e)}")
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": f"Error: {str(e)}", 
            "sequence": sequence
        })
