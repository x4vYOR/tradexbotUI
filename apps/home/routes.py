# -*- encoding: utf-8 -*-

# home/routes.py

from sunau import AUDIO_UNKNOWN_SIZE
from ..authentication.models import PortfolioMetric, User, Indicator, Condition, Pair, BuyStrategy, ModelMetric, Algorithm
from ..authentication.crud import get_current_user

from fastapi import APIRouter, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from ..database import get_db
from sqlalchemy.orm import Session
import ast

router = APIRouter()

templates = Jinja2Templates(directory="apps/templates")


@router.get("/index", response_class=HTMLResponse)
async def index(request: Request, user: User = Depends(get_current_user)):

    return templates.TemplateResponse(
        "home/index.html",
        {"request": request, "current_user": user, "segment": "index"},
    )

@router.get("/strategy", response_class=HTMLResponse)
async def strategy(request: Request, user: User = Depends(get_current_user)):

    return templates.TemplateResponse(
        "home/strategy/index.html",
        {"request": request, "current_user": user, "segment": "strategy"},
    )

@router.get("/strategy/new", response_class=HTMLResponse)
async def new_strategy(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    indicators = db.query(Indicator).order_by(Indicator.name).all()
    conditions = db.query(Condition).order_by(Condition.name).all()
    pairs = db.query(Pair).order_by(Pair.name).all()
    metrics = db.query(PortfolioMetric).order_by(PortfolioMetric.name).all()
    buy_strategies = db.query(BuyStrategy).order_by(BuyStrategy.name).all()
    aux_buy_strategies = []
    for item in buy_strategies:
        aux_buy_strategies.append({"id":item.id, "name": item.name+" - "+item.description})
    buy_strategies = aux_buy_strategies
    return templates.TemplateResponse(
        "home/strategy/new.html",
        {"request": request, "current_user": user, "segment": "strategy", "indicators": indicators, "conditions": conditions, "pairs": pairs, "metrics": metrics, "buy_strategies":buy_strategies},
    )

@router.get("/trainer", response_class=HTMLResponse)
async def trainer(request: Request, user: User = Depends(get_current_user)):

    return templates.TemplateResponse(
        "home/trainer/index.html",
        {"request": request, "current_user": user, "segment": "trainer"},
    )

@router.get("/trainer/new", response_class=HTMLResponse)
async def new_trainer(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    model_metrics = db.query(ModelMetric).order_by(ModelMetric.name).all()
    algorithms = db.query(Algorithm).order_by(Algorithm.name).all()
    aux_algorithms = []
    for item in algorithms:
        aux_algorithms.append({"id":item.id, "name": item.name, "parameters": ast.literal_eval(item.parameters)})
    algorithms = aux_algorithms
    return templates.TemplateResponse(
        "home/trainer/new.html",
        {"request": request, "current_user": user, "segment": "trainer", "model_metrics": model_metrics, "algorithms": algorithms},
    )

@router.get("/{template}", response_class=HTMLResponse)
async def route_template(
    request: Request, template: str, user: User = Depends(get_current_user)
):

    if not template.endswith(".html"):
        template += ".html"

    # Detect the current page
    segment = get_segment(request)

    # Serve the file (if exists) from app/templates/home/FILE.html
    return templates.TemplateResponse(
        f"home/{template}",
        {"request": request, "current_user": user, "segment": segment},
    )


# Helper - Extract current page name from request
def get_segment(request):
    try:
        segment = request.url.path.split("/")[-1]

        if segment == "":
            segment = "index"

        return segment

    except:
        return None
