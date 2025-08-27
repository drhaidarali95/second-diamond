from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

app = FastAPI()

# Serve static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve templates (HTML)
templates = Jinja2Templates(directory="templates")

# 🔑 Model setup (Gemma or fallback)
model_id = "google/gemma-2b-it"
hf_token = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu",
    token=hf_token
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat(message: str = Form(...)):
    inputs = tokenizer(message, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return JSONResponse({"markdown": response})

