# -*- encoding: utf-8 -*-

# home/routes.py

from calendar import c
from time import sleep
from fastapi import APIRouter, Depends
from xgboost import train
from ..authentication.crud import get_current_user
from ..authentication.models import AlgorithmDetail, PortfolioMetric, Bot,TrainSetup, Train,BacktestMetric,TrainModel,TrainMetric,Backtest,BacktestChart,PortfolioMetric, TradingSetup, TrainingDetail, TrainingMetric, User, Indicator, Condition, Pair, BuyStrategy, ModelMetric, Algorithm, StrategyPair,StrategyMetric, Rule
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from sse_starlette.sse import EventSourceResponse
from ..database import get_db
from .helpers import tuples_to_dict, extract_indicators
from sqlalchemy.orm import Session
import hashlib as hash
import requests
import ast
from ..train.prepareData import PrepareData
from ..train.Trainer import Trainer
import asyncio

router = APIRouter()

templates = Jinja2Templates(directory="apps/templates")

HOST = "http://35.188.143.189"
############ Train Results ############

@router.post("/train/run/all",response_class=EventSourceResponse)
async def train_run(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    form_data = await request.form()
    configs = tuples_to_dict(form_data.multi_items())
    configs = ast.literal_eval(configs["configs"])
    config_params = ast.literal_eval(configs[0]["parameter"])
    objPrepareData = PrepareData(config_params)
    objTrainer = Trainer(config_params, objPrepareData.train_dataset, objPrepareData.train_columns, objPrepareData.backtest_dataset, y_columns="target")
    async def event_generator():
        for item in configs:
            res = []
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break
            
            trained = objTrainer.train(ast.literal_eval(item["parameter"]))
            print("######## TRAINED")
            print(trained)
            # actualizando train
            db.query(Train).filter(Train.id == item["id"]).update(
                {
                    Train.status: "success",
                    Train.completed: True
                }
            )
            db.commit()
            # registrar los modelos
            for train in trained:
                trainmodel = TrainModel(
                    train_id = item["id"],
                    model = train["model"]["file"],
                    scaler = train["scaler"],
                    model_best_parameters = str(train["model"]["model_best_parameters"])
                )
                db.add(trainmodel)
                db.commit()
                db.refresh(trainmodel)
                #registrar los metrics del modelo
                for metric in train["model"]["metrics"]:
                    modelmetric = db.query(ModelMetric).filter(ModelMetric.name == metric["name"]).first()
                    trainmetric = TrainMetric(
                        value = metric["value"],
                        train_model_id = trainmodel.id,
                        model_metric_id = modelmetric.id
                    )
                    db.add(trainmetric)
                    db.commit()
                # registrar los backtest del modelo por par
                for backtest in train["backtest"]:
                    back = Backtest(
                        train_model_id = trainmodel.id,
                        pair = backtest["pair"]
                    )
                    db.add(back)
                    db.commit()
                    db.refresh(back)
                    # añadir grafica de velas
                    backchartcandle = BacktestChart(
                        name = "Chart candle",
                        path = backtest["chart_candles"],
                        backtest_id = back.id
                    )
                    db.add(backchartcandle)
                    db.commit()
                    # AÑADIR GRAFICA DE FONDOS
                    backchartfunds = BacktestChart(
                        name = "Chart Funds",
                        path = backtest["chart_funds"],
                        backtest_id = back.id
                    )
                    db.add(backchartfunds)
                    db.commit()
                    # añadir metricas del backtest
                    for metric in backtest["metrics"]:
                        portfoliometric = db.query(PortfolioMetric).filter(PortfolioMetric.name == metric["name"]).first()
                        backtestmetric = BacktestMetric(
                            value = metric["value"],
                            backtest_id = back.id,
                            portfolio_metric_id = portfoliometric.id
                        )
                    db.add(backtestmetric)
                    db.commit()
            
            
            train_setup = db.query(TrainSetup).filter(TrainSetup.id == ast.literal_eval(item["parameter"])["id"]).first()
            
            res = [{"id":item.id, "status": item.status, "checksum":item.checksum} for item in train_setup.trainings]
            #print(str(res)+"\n\n")
            yield str(res)

            await asyncio.sleep(3)
        
    return EventSourceResponse(event_generator())

@router.post("/train/proof", response_class=EventSourceResponse)
async def train_proof(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    form_data = await request.form()
    configs = tuples_to_dict(form_data.multi_items())
    configs = ast.literal_eval(configs["configs"])
    config_params = ast.literal_eval(configs[0]["parameter"])
    print(config_params)
    async def event_generator():
        value= {"id":10,"name":"RF", "values": [1,2,3]}
        event = "update"
        while True:
            # If client closes connection, stop sending events
            if await request.is_disconnected():
                break

            # Checks for new messages and return them to client if any
            
            yield str(value)

            await asyncio.sleep(3)

    return EventSourceResponse(event_generator())

############ Train Results ############

@router.get("/train/results/all/{id}",response_class=HTMLResponse)
async def train_results(id: int,request: Request,user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    train_setup = db.query(TrainSetup).filter(TrainSetup.id == id, TrainSetup.user_id == user.id).first()
    print(train_setup)
    print(train_setup.trainings)
    model_metrics = db.query(ModelMetric).all()
    return templates.TemplateResponse(
        "home/autotrain/results.html",
        {"request": request, 
        "current_user": user, 
        "model_metrics": model_metrics, 
        "trains": train_setup.trainings, 
        "train": train_setup.trainings[0].train_models[0], 
        "segment": "autotrain"
        },
    )

@router.get("/train/favorites",response_class=HTMLResponse)
async def train_favorites(request: Request,user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    train_favorites = db.query(TrainModel).join(Train).join(TrainSetup).filter(TrainModel.favorite == True, TrainSetup.user_id == user.id).all()
    print(train_favorites)
    model_metrics = db.query(ModelMetric).all()
    return templates.TemplateResponse(
        "home/autotrain/favorites.html",
        {"request": request, 
        "current_user": user, 
        "model_metrics": model_metrics, 
        "trains": train_favorites,
        "segment": "favorite"
        },
    )

@router.get("/train/backtest/{training_id}/{pair_name}/{model_name}",response_class=JSONResponse)
async def backtest_results(training_id: int,pair_name: str,model_name: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    train = db.query(Train).join(TrainSetup).filter(Train.id == training_id, TrainSetup.user_id == user.id).first()
    print(train)
    train_model = db.query(TrainModel).filter(TrainModel.model.like('%'+model_name+'%')).filter(TrainModel.train_id == train.id).first()
    print(train_model)
    backtest = db.query(Backtest).filter(Backtest.train_model_id == train_model.id, Backtest.pair == pair_name).first()
    print(backtest)
    chart_candle = db.query(BacktestChart).filter(BacktestChart.name == "Chart candle", BacktestChart.backtest_id == backtest.id).first()
    chart_funds = db.query(BacktestChart).filter(BacktestChart.name == "Chart Funds", BacktestChart.backtest_id == backtest.id).first()
    metrics = [{"name": item.portfolio_metric.name, "value": item.value} for item in backtest.backtest_metrics]

    return JSONResponse(content={"chart_candle":chart_candle.path[1:], "chart_funds": chart_funds.path[1:], "metrics": metrics, "favorite":train_model.favorite},status_code=200)

@router.get("/train/favorite/{training_id}/{model_name}", response_class=JSONResponse)
async def set_favorite(training_id: int,model_name: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        train = db.query(Train).join(TrainSetup).filter(Train.id == training_id, TrainSetup.user_id == user.id).first()
        train_model = db.query(TrainModel).filter(TrainModel.model.like('%'+model_name+'%')).filter(TrainModel.train_id == train.id).first()
        print(train_model.favorite)
        new_val = False if train_model.favorite else True
        db.query(TrainModel).filter(TrainModel.id == train_model.id).update(
            {TrainModel.favorite: new_val
            }
        )
        db.commit()        
        return JSONResponse(content={"success":True}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={"success":False}, status_code=201)

@router.get("/favorite/remove/{training_id}", response_class=JSONResponse)
async def remove_favorite(training_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        db.query(TrainModel).filter(TrainModel.id == training_id).update(
            {TrainModel.favorite: False
            }
        )
        db.commit()        
        return JSONResponse(content={"success":True}, status_code=200)
    except Exception as e:
        print(e)
        return JSONResponse(content={"success":False}, status_code=201)

@router.get("/favorite/backtest/{trainmodel_id}/{pair_name}",response_class=JSONResponse)
async def favorite_results(trainmodel_id: int, pair_name: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    train_model = db.query(TrainModel).join(Train).join(TrainSetup).filter(TrainModel.favorite == True, TrainModel.id == trainmodel_id, TrainSetup.user_id == user.id).first()
    print(train_model)
    if pair_name != "name":
        backtest = db.query(Backtest).filter(Backtest.train_model_id == train_model.id, Backtest.pair == pair_name).first()
    else:
        backtest = db.query(Backtest).filter(Backtest.train_model_id == train_model.id).first()
    pairs = [item.pair for item in train_model.backtests]
    print(backtest)
    chart_candle = db.query(BacktestChart).filter(BacktestChart.name == "Chart candle", BacktestChart.backtest_id == backtest.id).first()
    chart_funds = db.query(BacktestChart).filter(BacktestChart.name == "Chart Funds", BacktestChart.backtest_id == backtest.id).first()
    metrics = [{"name": item.portfolio_metric.name, "value": item.value} for item in backtest.backtest_metrics]

    return JSONResponse(content={"chart_candle":chart_candle.path[1:], "chart_funds": chart_funds.path[1:], "metrics": metrics, "pairs": pairs, "id":train_model.id, "model_name": train_model.model.split('/')[2].split("_")[0]},status_code=200)

############## Autotrade #####################

@router.post("/autotrade/add", response_class=JSONResponse)
async def autotrade_add(
        request: Request, user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
    try:
        form_data = await request.form()
        form_data = tuples_to_dict(form_data.multi_items())
        print(form_data)
        new_bot = Bot(
            name = form_data["name"],
            train_model_id = form_data["id"],
            status = "stopped"
        )
        db.add(new_bot)
        db.commit()

        return JSONResponse(content={"success":True},status_code=200)
    except Exception as e:
        return JSONResponse(content={"success":False, "message":e},status_code=401)

@router.get("/autotrade/list", response_class=HTMLResponse)
async def autotrade_list(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    bots = db.query(Bot).join(TrainModel).join(Train).join(TrainSetup).filter(TrainSetup.user_id == user.id).order_by(Bot.name).all()
    return templates.TemplateResponse(
        "home/trader/index.html",
        {"request": request, "current_user": user, "segment": "bot", "bots": bots},
    )

@router.post("/autotrade/run", response_class=JSONResponse)
async def autotrade_run(
        request: Request, user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
    try:
        form_data = await request.form()
        form_data = tuples_to_dict(form_data.multi_items())
        bot = db.query(Bot).join(TrainModel).join(Train).join(TrainSetup).filter(Bot.id == int(form_data["id"]), TrainSetup.user_id == user.id).first()
    
        res = requests.post(HOST+'/api/login',json={"username": "x4vyjm", "email": "x4vyjm@gmail.com"})
        token = res.text.replace('"','')
        headers = {"Authorization":"Bearer "+token}
        data = {
                "config": "config"
        }
        response = requests.post(HOST+'/api/data', json=data, headers=headers)
        
        if(bot.status == "stopped"):
            pass
            #connect to api

            # call api and check if bot already exist, send bot uuid 

            # if exist send uuid and command run

            # else send train config, uuid to create and run

            # call api and run bot. send train config, bot uuid, bot id, command (run)

            db.query(Bot).filter(Bot.id == bot.id).update(
                {
                    Bot.status: "running"
                }
            )
            db.commit()
        else:
            pass
            # call api and stop bot. send bot uuid
            db.query(Bot).filter(Bot.id == bot.id).update(
                {
                    Bot.status: "stopped"
                }
            )
            db.commit()


        return JSONResponse(content={"success": True},status_code=200)
    except Exception as e:
        return JSONResponse(content={"success": False, "message":e},status_code=401)

"""
config = jsonable_encoder(item["config"])
config = ast.literal_eval(config)
for item in list(form_data):
    item = tuples_to_dict(item.multi_items())
    
    config = jsonable_encoder(item["config"])
    config = ast.literal_eval(config)
    
    objPrepareData = PrepareData(config)

    objTrainer = Tra


    indicators = extract_indicators(config["train_indicators"],config["strategy"]["rules"])
    models = config["training_detail"]["algorithm_details"]
    pairs = [item["pair"]["name"] for item in config["strategy"]["strategy_pairs"]]
    print(type(models))
    print(indicators)
    print(models)
    #print(config["training_detail"])
"""
    
    

