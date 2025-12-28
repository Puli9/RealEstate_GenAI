from dotenv import load_dotenv
load_dotenv()

import os
from typing import Dict, Any

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.stage3 import run_stage3_pipeline


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def form_get(request: Request):
    return templates.TemplateResponse(
        "form.html",
        {"request": request, "narrative": None, "prediction": None},
    )

@app.post("/", response_class=HTMLResponse)
async def form_post(
    request: Request,
    city: str = Form(...),
    locality: str = Form(...),
    property_type: str = Form(...),
    size_sqft: float = Form(...),
    age_yrs: float = Form(...),
    user_ctx: str = Form(...),
):
    property_input: Dict[str, Any] = {
        "city": city,
        "locality": locality,
        "property_type": property_type,
        "size_sqft": size_sqft,
        "age_yrs": age_yrs,
    }

    result = run_stage3_pipeline(property_input, user_ctx)
    narrative = result["narrative"]
    prediction = result["prediction"]

    return templates.TemplateResponse(
        "form.html",
        {
            "request": request,
            "narrative": narrative,
            "prediction": prediction,
        },
    )
