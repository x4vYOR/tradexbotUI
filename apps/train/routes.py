# -*- encoding: utf-8 -*-

# home/routes.py
from fastapi import APIRouter, Depends
from ..authentication.crud import get_current_user
from ..api import get_api
from ..authentication.models import AlgorithmDetail, PortfolioMetric, Bot,TrainSetup, Train,BacktestMetric,TrainModel,TrainMetric,Backtest,BacktestChart,PortfolioMetric, TradingSetup, TrainingDetail, TrainingMetric, User, Indicator, Condition, Pair, BuyStrategy, ModelMetric, Algorithm, StrategyPair,StrategyMetric, Rule
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from sse_starlette.sse import EventSourceResponse
from ..database import get_db
from .helpers import tuples_to_dict, extract_indicators, unixtodate
from sqlalchemy.orm import Session
import requests
import ast
from ..train.prepareData import PrepareData
from ..train.Trainer import Trainer
import json
import asyncio
import datetime
import pandas as pd

router = APIRouter()

templates = Jinja2Templates(directory="apps/templates")

############ Train Results ############

@router.post("/train/run/all",response_class=EventSourceResponse)
async def train_run(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db), api: dict = Depends(get_api)):
    form_data = await request.form()
    configs = tuples_to_dict(form_data.multi_items())
    configs = ast.literal_eval(configs["configs"])
    config_params = ast.literal_eval(configs[0]["parameter"])
    objPrepareData = PrepareData(config_params, api)
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
                    columns = str(objPrepareData.train_columns),
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
    print(train_setup.trainings[0])
    print(train_setup.trainings[0].train_models)
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
    print(len(train_model.bots))
    has_bot = True if len(train_model.bots)>0 else False
    chart_candle = db.query(BacktestChart).filter(BacktestChart.name == "Chart candle", BacktestChart.backtest_id == backtest.id).first()
    chart_funds = db.query(BacktestChart).filter(BacktestChart.name == "Chart Funds", BacktestChart.backtest_id == backtest.id).first()
    metrics = [{"name": item.portfolio_metric.name, "value": item.value} for item in backtest.backtest_metrics]

    return JSONResponse(content={"chart_candle":chart_candle.path[1:], "chart_funds": chart_funds.path[1:], "metrics": metrics, "pairs": pairs, "id":train_model.id, "model_name": train_model.model.split('/')[2].split("_")[0], "has_bot":has_bot},status_code=200)

############## Autotrade #####################

@router.post("/autotrade/add", response_class=JSONResponse)
async def autotrade_add(
        request: Request, user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
        api: dict = Depends(get_api)
    ):
    try:
        print("HOLA")
        form_data = await request.form()
        form_data = tuples_to_dict(form_data.multi_items())
        print(f"form_data: {form_data} type {form_data} ")
        # register new bot in API tradexbot
        train_model = db.query(TrainModel).filter(TrainModel.id == int(form_data["id"])).first()
        print("HOLA 1")
        config_params = ast.literal_eval(train_model.train.parameters)
        print("HOLA2")
        aux_columns = ast.literal_eval(train_model.columns.replace(" ",""))
        print("HOLA 3")
        aux_pairs = [item["pair"]["name"] for item in config_params["strategy"]["strategy_pairs"]]
        # upload SCALER
        files = {'file': open(train_model.scaler, 'rb')}
        response = requests.post(api["host"]+"/api/upload", files=files, headers=api["headers"])
        res_scaler = json.loads(response.content)
        files["file"].close()
        if res_scaler["message"] == "Error":
            return JSONResponse(content={"success":False, "message":"Scaler file not uploaded"},status_code=201)
        # upload MODEL
        files = {'file': open(train_model.model, 'rb')}
        response = requests.post(api["host"]+"/api/upload", files=files, headers=api["headers"])
        res_model = json.loads(response.content)
        files["file"].close()
        if res_model["message"] == "Error":
            return JSONResponse(content={"success":False, "message":"Model file not uploaded"},status_code=201)
        #register BOT
        query = {
            "columns": aux_columns,
            "pairs": aux_pairs,
            "scaler": res_scaler["filename"],
            "model": res_model["filename"],
            "timeframe": config_params["timeframe"],
            "api_key": config_params["trading_setup"]["api_key"],
            "api_secret": config_params["trading_setup"]["api_secret"],
            "exchange": config_params["trading_setup"]["exchange"],
            "initial_capital": float(config_params["trading_setup"]["capital"])/len(aux_pairs),
            "capital_pair": {item:float(config_params["trading_setup"]["capital"])/len(aux_pairs) for item in aux_pairs},
            "currency_base": config_params["trading_setup"]["currency_base"],
            "strategy": config_params["strategy"]["buy_strategy"]["name"],
            "strategy_parameters": str(config_params["strategy"]["strategy_parameters"])
        }
        print(query)
        response = requests.post(api["host"]+'/api/new/bot', json=query, headers=api["headers"])
        res = json.loads(response.content)
        print(res)
        #register bot internally
        if(res["message"] == "Success"):
            # si se registro el bot agregalo en la bd local
            new_bot = Bot(
                name = form_data["name"],
                train_model_id = form_data["id"],
                uuid = res["uuid"],
                status = "stopped"
            )
            db.add(new_bot)
            db.commit()
            return JSONResponse(content={"success":True},status_code=200)
        else:
            return JSONResponse(content={"success":False, "message":"Not registered"},status_code=201)
    except Exception as e:
        print(e)
        return JSONResponse(content={"success":False, "message":"Action Failed"},status_code=401)

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
        db: Session = Depends(get_db), api: dict = Depends(get_api)
    ):
    try:
        form_data = await request.form()
        form_data = tuples_to_dict(form_data.multi_items())
        bot = db.query(Bot).join(TrainModel).join(Train).join(TrainSetup).filter(Bot.id == int(form_data["id"]), Bot.uuid == str(form_data["uuid"]), TrainSetup.user_id == user.id).first()
        data = {
                "uuid": bot.uuid
        }      
        print("data dict ", data)  
        if(bot.status == "stopped"):
            #connect to api and run
            # if exist send uuid            
            response = requests.post(api["host"]+'/api/run/bot', json=data, headers=api["headers"])
            res = json.loads(response.content)
            print(res)
            print(res["message"])
            print(res["task_id"])
            # if success running, change status and set task_id
            if(res["message"] == "Success"):
                db.query(Bot).filter(Bot.id == bot.id).update(
                    {
                        Bot.status: "running",
                        Bot.task_id: str(res["task_id"])
                    }
                )
                db.commit()
                print("success bot register")
                return JSONResponse(content={"success": True, "id":bot.id, "uuid":bot.uuid, "status":"running", "task_id": str(res["task_id"])},status_code=200)
            else:
                print(res["error"])
                return JSONResponse(content={"success": False, "message": res["error"]},status_code=200)
        else:
            # call api and stop bot. send bot uuid
            print("task_id: ",bot.task_id)
            data["task_id"] = bot.task_id
            print("data dict ", data)  
            response = requests.post(api["host"]+'/api/stop/bot', json=data, headers=api["headers"])
            res = json.loads(response.content)
            print(res)
            if(res["message"] == "Success"): #res["message"] == "Success"
                #if succes stopped change bot status and delete its task_id
                db.query(Bot).filter(Bot.id == bot.id).update(
                    {
                        Bot.status: "stopped",
                        Bot.task_id: ""
                    }
                )
                db.commit()
                print("changed status to stopped")
                return JSONResponse(content={"success": True, "id":bot.id, "uuid":bot.uuid, "status":"stopped", "task_id": ""},status_code=200)
            else:
                print(res["error"])
                return JSONResponse(content={"success": False, "message": res["error"]},status_code=200)

    except Exception as e:
        return JSONResponse(content={"success": False, "message":e},status_code=401)

@router.get("/autotrade/results/{bot_id}/{bot_uuid}",response_class=HTMLResponse)
async def autotrade_results(bot_id: int, bot_uuid: str,request: Request,user: User = Depends(get_current_user), db: Session = Depends(get_db), api: dict = Depends(get_api)):
    bot = db.query(Bot).filter(Bot.id == bot_id, Bot.uuid == bot_uuid).first()
    config_params = ast.literal_eval(bot.train_model.train.parameters)
    print("HOLA 3")
    pairs = [item["pair"]["name"] for item in config_params["strategy"]["strategy_pairs"]]
    # get trades of bot from api
    data = {
        "uuid": bot.uuid
    }
    print(data)
    response = requests.post(api["host"]+'/api/trades/bot', json=data, headers=api["headers"])
    res = json.loads(response.content)
    print(res)
    date_evolution = []
    portfolio_evolution = []
    invested_evolution = []
    available_evolution = []
    trades = []
    if(res["message"] == "Success"):
        #Extract pairs
        trades = res["trades"]
        df_capital = pd.DataFrame.from_dict(res["capital"]) # close_time, portfolio, invested, available
        df_capital["close_time"] = pd.to_datetime(df_capital["close_time"], unit="ms")
        print("df_capital: ", df_capital)
        #df_capital["close_time"] = df_capital.close_time.astype('datetime64[ns]')
        #print("df_capital: ", df_capital)
        #split trades by pairs 
        #print(f"trades: {res['trades']}")
        #print(f"capital: {res['capital']}")
        aux_date_ini = bot.created_at - datetime.timedelta(days=1)
        aux_date_end = datetime.datetime.now() + datetime.timedelta(days=1)
        #extract date range and get data
        data = {
            "pair": str(res["trades"][0]["pair"]),
            "timeframe": config_params["timeframe"],
            "ini": str(aux_date_ini.strftime("%d/%m/%Y")),
            "end": str(aux_date_end.strftime("%d/%m/%Y")),
            "indicators": ['open_time','close_time']
        }
        print(data)
        r = requests.post(api["host"]+'/api/data', json=data, headers=api["headers"])
        #print("data: ",r.content)
        df = pd.read_json(r.json()["data"])
        #print(df)
        df = df.dropna() # df [close_time, open_time]
        #df["close_time"] = df.close_time.astype('datetime64[ns]')
        print("df: ", df)
        result_df = pd.merge(df, df_capital, how="left", on=["close_time"])
        result_df.fillna(method='ffill', inplace=True)

        print("result_df: ", result_df)
        result_df["open_time"] = pd.to_datetime(result_df["open_time"], unit="ms")
        result_df["close_time"] = result_df.close_time.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
        result_df = result_df.fillna(0)
        # convert unix dates to readable
        for trade in trades:
            trade["open_date"] = unixtodate(trade["open_date"])
            trade["close_date"] = unixtodate(trade["close_date"])
            trade["close_candle"] = unixtodate(trade["close_candle"])
        # ensemble portfolio evolution chart
        date_evolution = result_df["close_time"].tolist()
        portfolio_evolution = result_df["portfolio"].tolist()
        invested_evolution = result_df["invested"].tolist()
        available_evolution = result_df["available"].tolist()
    
    return templates.TemplateResponse(
        "home/trader/results.html",
        {"request": request, 
        "current_user": user, 
        "pairs": pairs, 
        "trades": trades, 
        "portfolio_evolution":portfolio_evolution,
        "date_evolution":date_evolution,
        "invested_evolution":invested_evolution,
        "available_evolution":available_evolution,
        "bot": bot, 
        "segment": "bot"
        },
    )

@router.get("/autotrade/result/{bot_id}/{bot_uuid}/{pair}",response_class=JSONResponse)
async def autotrade_pair_results(bot_id: int, bot_uuid: str, pair: str, request: Request,user: User = Depends(get_current_user), db: Session = Depends(get_db), api: dict = Depends(get_api)):
    bot = db.query(Bot).filter(Bot.id == bot_id, Bot.uuid == bot_uuid).first()
    config_params = ast.literal_eval(bot.train_model.train.parameters)
    # get trades of bot from api
    data = {
        "uuid": bot.uuid,
        "pair":pair
    }
    response = requests.post(api["host"]+'/api/trades/pair', json=data, headers=api["headers"])
    res = json.loads(response.content)
    print(res)
    if(res["message"] == "Success"):
        trades = res['trades']
        #df_trades = pd.DataFrame.from_dict(res["trades"]) # 
        #df_trades.rename(columns = {'close_time':'close_data'}, inplace = True)
        #df_trades.rename(columns = {'close_candle':'close_time'}, inplace = True)
        #print(df_trades)
        #split trades by pairs 
        print(f"trades: {res['trades']}")
        aux_date_ini = bot.created_at - datetime.timedelta(days=1)
        aux_date_end = datetime.datetime.now() + datetime.timedelta(days=1)      
        #extract date range and get data
        data = {
            "pair": pair,
            "timeframe": config_params["timeframe"],
            "ini": str(aux_date_ini.strftime("%d/%m/%Y")),
            "end": str(aux_date_end.strftime("%d/%m/%Y")),
            "indicators": ['open_time','close_time','open','high','low','close']
        }
        print(data)
        r = requests.post(api["host"]+'/api/data', json=data, headers=api["headers"])
        df = pd.read_json(r.json()["data"])
        #print(df)
        df = df.dropna() # df [close_time, open_time]

        #df_trades["close_time"] = pd.to_datetime(df_trades["close_time"], unit="ms")
        #df_trades["close_date"] = pd.to_datetime(df_trades["close_date"], unit="ms")
        #df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        #df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        df["close_time"] = df.close_time.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
        df["open_time"] = df.open_time.apply(lambda x : (x-datetime.datetime(1970,1,1)).total_seconds())
        print(df)
        #result_df = pd.merge(df, df_trades["close_time","entry_price","close_price","close_date"], how="left", on=["close_time"])
        #result_df.fillna(method='ffill', inplace=True)
        #print("result_df: ", result_df)
        #for trade in trades:
        #    trade["open_date"] = unixtodate(trade["open_date"])
        #    trade["close_date"] = unixtodate(trade["close_date"])
        #    trade["close_candle"] = unixtodate(trade["close_candle"])
        print("##############################")
        print(trades)
        return JSONResponse(content={ 
            "success": True, 
            "data":{
                "open_time":str(df["open_time"].tolist()),
                "close_time": str(df["close_time"].tolist()),
                "open": df["open"].tolist(),
                "high": df["high"].tolist(),
                "low": df["low"].tolist(),
                "close": df["close"].tolist()
            },
            "trades": trades },status_code=200)
    return JSONResponse(content={"success": False, "message": res["error"]},status_code=200)
"""
,
            "trades_data":{
                "close_time": df_trades["close_time"].tolist(),
                "close_date": df_trades["close_date"].tolist(),
                "entry_price": df_trades["entry_price"].tolist(),
                "close_price": df_trades["close_price"].tolist(),

            }
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
    
    

