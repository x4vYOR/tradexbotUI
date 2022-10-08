# -*- encoding: utf-8 -*-

# home/routes.py

from ..authentication.models import AlgorithmDetail, PortfolioMetric, Strategy, TrainIndicator,TrainSetup, TradingSetup, TrainingDetail, TrainingMetric, User, Indicator, Condition, Pair, BuyStrategy, ModelMetric, Algorithm, StrategyPair,StrategyMetric, Rule
from ..authentication.crud import get_current_user

from fastapi import APIRouter, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.requests import Request
from ..database import get_db
from sqlalchemy.orm import Session
import ast
from .helpers import tuples_to_dict
router = APIRouter()

templates = Jinja2Templates(directory="apps/templates")


@router.get("/index", response_class=HTMLResponse)
async def index(request: Request, user: User = Depends(get_current_user)):

    return templates.TemplateResponse(
        "home/index.html",
        {"request": request, "current_user": user, "segment": "index"},
    )

############# AutoTrain ################

@router.get("/autotrain", response_class=HTMLResponse)
async def autotrain(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    trainings = db.query(TrainSetup).order_by(TrainSetup.created_at).all()
    
    return templates.TemplateResponse(
        "home/autotrain/index.html",
        {"request": request, 
        "current_user": user, 
        "segment": "autotrain", 
        "trainings":trainings
        },
    )

@router.get("/autotrain/new", response_class=HTMLResponse)
async def new_autotrain(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    strategies = db.query(Strategy.id, Strategy.name, Strategy.description).order_by(Strategy.created_at).all()
    training_details = db.query(TrainingDetail.id, TrainingDetail.name, TrainingDetail.description).order_by(TrainingDetail.created_at).all()
    trading_settings = db.query(TradingSetup.id, TradingSetup.name, TradingSetup.description).order_by(TradingSetup.created_at).all()
    indicators = db.query(Indicator).order_by(Indicator.name).all()
    return templates.TemplateResponse(
        "home/autotrain/new.html",
        {"request": request,
            "current_user": user, 
            "segment": "autotrain",
            "indicators":indicators,
            "strategies":strategies,
            "training_details": training_details,
            "trading_settings":trading_settings
        }
    )

@router.post("/autotrain/new", response_class=RedirectResponse)
async def create_autotrain(
        request: Request, user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
    form_data = await request.form()
    form_data = tuples_to_dict(form_data.multi_items())
    print(form_data)
    autotrain = TrainSetup(
                    name = form_data["name"],
                    description = form_data["description"],
                    timeframe = form_data["timeframe"],
                    train_data_start = form_data["train_data"].split(" - ")[0],
                    train_data_end = form_data["train_data"].split(" - ")[1],
                    backtest_data_start = form_data["backtest_data"].split(" - ")[0],
                    backtest_data_end = form_data["backtest_data"].split(" - ")[1],
                    user_id = user.id,
                    trading_setup_id = form_data["trading"],
                    training_detail_id = form_data["training"],
                    strategy_id = form_data["strategy"]
                )
    db.add(autotrain)
    db.commit()
    db.refresh(autotrain)
    
    train_indicators = [TrainIndicator(
        train_setup_id = autotrain.id,
        indicator_id = item
    ) for item in form_data["indicators[]"]]
    db.add_all(train_indicators)
    db.commit()
    return RedirectResponse("/autotrain",status_code=303)

@router.get("/autotrain/edit/{id_autotrain}", response_class=HTMLResponse)
async def edit_autotrain(request: Request,id_autotrain: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    training = db.query(TrainSetup).filter(TrainSetup.id==id_autotrain).first()
    strategies = db.query(Strategy.id, Strategy.name, Strategy.description).order_by(Strategy.created_at).all()
    training_details = db.query(TrainingDetail.id, TrainingDetail.name, TrainingDetail.description).order_by(TrainingDetail.created_at).all()
    trading_settings = db.query(TradingSetup.id, TradingSetup.name, TradingSetup.description).order_by(TradingSetup.created_at).all()
    indicators = db.query(Indicator).order_by(Indicator.name).all()
    autotrain_indicators = [item.indicator_id for item in training.train_indicators]
    return templates.TemplateResponse(
        "home/autotrain/edit.html",
        {"request": request,
            "current_user": user, 
            "segment": "autotrain",
            "autotrain_indicators":autotrain_indicators,
            "indicators":indicators,
            "training":training,
            "strategies":strategies,
            "training_details": training_details,
            "trading_settings":trading_settings
        }
    )

@router.post("/autotrain/edit", response_class=HTMLResponse)
async def update_autotrain(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    form_data = await request.form()
    form_data = tuples_to_dict(form_data.multi_items())
    print(form_data)    
    db.query(TrainSetup).filter(TrainSetup.id == form_data["training_id"]).update(
        {
            TrainSetup.name:form_data["name"],
            TrainSetup.description:form_data["description"],
            TrainSetup.timeframe:form_data["timeframe"],
            TrainSetup.train_data_start:form_data["train_data"].split(" - ")[0],
            TrainSetup.train_data_end:form_data["train_data"].split(" - ")[1],
            TrainSetup.backtest_data_start:form_data["backtest_data"].split(" - ")[0],
            TrainSetup.backtest_data_end:form_data["backtest_data"].split(" - ")[1],
            TrainSetup.user_id:user.id,
            TrainSetup.trading_setup_id:form_data["trading"],
            TrainSetup.training_detail_id:form_data["training"],
            TrainSetup.strategy_id:form_data["strategy"]
        })
    db.commit()

    training = db.query(TrainSetup).get(form_data["training_id"])

    for item in training.train_indicators:
        db.delete(item)
        db.commit()
    train_indicators = [TrainIndicator(
        train_setup_id = training.id,
        indicator_id = item
    ) for item in form_data["indicators[]"]]
    db.add_all(train_indicators)
    db.commit()

    return RedirectResponse("/autotrain",status_code=303)

############# Settings ################

@router.get("/settings", response_class=HTMLResponse)
async def strategy(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    strategies = db.query(Strategy).order_by(Strategy.created_at).all()
    training_details = db.query(TrainingDetail).order_by(TrainingDetail.created_at).all()
    trading_settings = db.query(TradingSetup).order_by(TradingSetup.created_at).all()
    
    return templates.TemplateResponse(
        "home/settings/index.html",
        {"request": request, 
        "current_user": user, 
        "segment": "settings", 
        "strategies":strategies,
        "training_details": training_details,
        "trading_settings":trading_settings
        },
    )

# ############# STRATEGY ################

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

@router.post("/strategy/new", response_class=HTMLResponse)
async def create_strategy(
        request: Request, user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
    form_data = await request.form()
    form_data = tuples_to_dict(form_data.multi_items())
    
    strategy = Strategy(
                    name = form_data["name"],
                    description = form_data["description"],
                    buy_strategy_id = form_data["buy_strategy"],
                    user_id = user.id,
                    json_parameters = str(form_data)
                )
    db.add(strategy)
    db.commit()
    db.refresh(strategy)

    form_data.pop("buy_strategy")

    strategy_pairs = [StrategyPair(
        strategy_id = strategy.id,
        pair_id = item
    ) for item in form_data["pairs[]"]]
    # Add user to database
    db.add_all(strategy_pairs)
    db.commit()

    strategy_metrics = [StrategyMetric(
        strategy_id = strategy.id,
        portfolio_metric_id = item
    ) for item in form_data["metrics[]"]]
    # Add user to database
    db.add_all(strategy_metrics)
    db.commit()

    buy_keys = [item for item in form_data.keys() if 'buy_' in item]
    buy_index = list(set([item.split("_")[2] for item in buy_keys]))
    sell_keys = [item for item in form_data.keys() if 'sell_' in item]
    sell_index = list(set([item.split("_")[2] for item in sell_keys]))

    buy_rules = []
    for index in buy_index:
        if "buy_valueinput_"+index in buy_keys:
            buy_rules.append({"first_indicator_id": form_data["buy_indicator_"+index], "condition_id": form_data["buy_condition_"+index], "second_indicator_id": form_data["buy_value_"+index], "value": form_data["buy_valueinput_"+index]})
        else:
            buy_rules.append({"first_indicator_id": form_data["buy_indicator_"+index], "condition_id": form_data["buy_condition_"+index], "second_indicator_id": form_data["buy_value_"+index], "value": 0})

    sell_rules = []
    for index in sell_index:
        if "sell_valueinput_"+index in sell_keys:
            sell_rules.append({"first_indicator_id": form_data["sell_indicator_"+index], "condition_id": form_data["sell_condition_"+index], "second_indicator_id": form_data["sell_value_"+index], "value": form_data["sell_valueinput_"+index]})
        else:
            sell_rules.append({"first_indicator_id": form_data["sell_indicator_"+index], "condition_id": form_data["sell_condition_"+index], "second_indicator_id": form_data["sell_value_"+index], "value": 0})

    buy_rules_strategy = [Rule(
        order_type = True,
        strategy_id = strategy.id,
        first_indicator_id = item["first_indicator_id"],
        second_indicator_id = item["second_indicator_id"],
        condition_id = item["condition_id"],
        value = item["value"]
    ) for item in buy_rules]
    # Add user to database
    db.add_all(buy_rules_strategy)
    db.commit()
    print(buy_rules)

    sell_rules_strategy = [Rule(
        order_type = False,
        strategy_id = strategy.id,
        first_indicator_id = item["first_indicator_id"],
        second_indicator_id = item["second_indicator_id"],
        condition_id = item["condition_id"],
        value = item["value"]
    ) for item in sell_rules]
    # Add user to database
    db.add_all(sell_rules_strategy)
    db.commit()

    backtest_keys = [item for item in form_data.keys() if 'backtest_' in item]
    filter_keys = [item for item in form_data.keys() if 'filter_' in item]
    target_keys = [item for item in form_data.keys() if 'target_' in item]

    parameters = {"backtest": {},"filter":{},"target":{}}
    for item in backtest_keys:
        parameters["backtest"][item] = form_data[item]
    for item in filter_keys:
        parameters["filter"][item] = form_data[item]
    for item in target_keys:
        parameters["target"][item] = form_data[item]

    #strategy.strategy_parameters = str(parameters)
    db.query(Strategy).filter(Strategy.id == strategy.id).update({Strategy.strategy_parameters:str(parameters)})
    db.commit()
    print(parameters)


    return RedirectResponse("/settings",status_code=303)

@router.get("/strategy/edit/{id_strat}", response_class=HTMLResponse)
async def edit_strategy(request: Request,id_strat: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    indicators = db.query(Indicator).order_by(Indicator.name).all()
    conditions = db.query(Condition).order_by(Condition.name).all()
    pairs = db.query(Pair).order_by(Pair.name).all()
    metrics = db.query(PortfolioMetric).order_by(PortfolioMetric.name).all()
    buy_strategies = db.query(BuyStrategy).order_by(BuyStrategy.name).all()
    strategy = db.query(Strategy).get(id_strat)
    strategy_pairs = [item.pair_id for item in strategy.strategy_pairs]
    print(strategy_pairs)
    strategy_metrics = [item.portfolio_metric_id for item in strategy.strategy_metrics]
    print(strategy_metrics)
    buy_rules = [item for item in strategy.rules if item.order_type==1]
    sell_rules = [item for item in strategy.rules if item.order_type==0]
    aux_buy_strategies = []
    for item in buy_strategies:
        aux_buy_strategies.append({"id":item.id, "name": item.name+" - "+item.description})
    buy_strategies = aux_buy_strategies
    return templates.TemplateResponse(
        "home/strategy/edit.html",
        {"request": request, "current_user": user, 
        "segment": "strategy", "strategy_pairs": strategy_pairs, 
        "strategy_metrics": strategy_metrics, "buy_rules": buy_rules, 
        "sell_rules": sell_rules,
        "strategy": strategy, "indicators": indicators, 
        "conditions": conditions, "pairs": pairs, 
        "metrics": metrics, "buy_strategies":buy_strategies},
    )

@router.post("/strategy/edit", response_class=HTMLResponse)
async def update_strategy(
        request: Request, user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
    form_data = await request.form()
    form_data = tuples_to_dict(form_data.multi_items())
    print(form_data)
    strategy = db.query(Strategy).get(form_data["strategy_id"])

    """
    strategy = Strategy(
                    name = form_data["name"],
                    description = form_data["description"],
                    buy_strategy_id = form_data["buy_strategy"],
                    user_id = user.id,
                    json_parameters = str(form_data)
                )
    db.add(strategy)
    db.commit()
    db.refresh(strategy)
    """
    aux_buy_strategy_id = form_data["buy_strategy"]
    form_data.pop("buy_strategy")
    for item in strategy.strategy_pairs:
        db.delete(item)
        db.commit()
    strategy_pairs = [StrategyPair(
        strategy_id = strategy.id,
        pair_id = item
    ) for item in form_data["pairs[]"]]
    # Add user to database
    db.add_all(strategy_pairs)
    db.commit()

    for item in strategy.strategy_metrics:
        db.delete(item)
        db.commit()
    strategy_metrics = [StrategyMetric(
        strategy_id = strategy.id,
        portfolio_metric_id = item
    ) for item in form_data["metrics[]"]]
    # Add user to database
    db.add_all(strategy_metrics)
    db.commit()

    for item in strategy.rules:
        db.delete(item)
        db.commit()
    buy_keys = [item for item in form_data.keys() if 'buy_' in item]
    buy_index = list(set([item.split("_")[2] for item in buy_keys]))
    sell_keys = [item for item in form_data.keys() if 'sell_' in item]
    sell_index = list(set([item.split("_")[2] for item in sell_keys]))

    buy_rules = []
    for index in buy_index:
        if "buy_valueinput_"+index in buy_keys:
            buy_rules.append({"first_indicator_id": form_data["buy_indicator_"+index], "condition_id": form_data["buy_condition_"+index], "second_indicator_id": form_data["buy_value_"+index], "value": form_data["buy_valueinput_"+index]})
        else:
            buy_rules.append({"first_indicator_id": form_data["buy_indicator_"+index], "condition_id": form_data["buy_condition_"+index], "second_indicator_id": form_data["buy_value_"+index], "value": 0})

    sell_rules = []
    for index in sell_index:
        if "sell_valueinput_"+index in sell_keys:
            sell_rules.append({"first_indicator_id": form_data["sell_indicator_"+index], "condition_id": form_data["sell_condition_"+index], "second_indicator_id": form_data["sell_value_"+index], "value": form_data["sell_valueinput_"+index]})
        else:
            sell_rules.append({"first_indicator_id": form_data["sell_indicator_"+index], "condition_id": form_data["sell_condition_"+index], "second_indicator_id": form_data["sell_value_"+index], "value": 0})

    buy_rules_strategy = [Rule(
        order_type = True,
        strategy_id = strategy.id,
        first_indicator_id = item["first_indicator_id"],
        second_indicator_id = item["second_indicator_id"],
        condition_id = item["condition_id"],
        value = item["value"]
    ) for item in buy_rules]
    # Add user to database
    db.add_all(buy_rules_strategy)
    db.commit()
    print(buy_rules)

    sell_rules_strategy = [Rule(
        order_type = False,
        strategy_id = strategy.id,
        first_indicator_id = item["first_indicator_id"],
        second_indicator_id = item["second_indicator_id"],
        condition_id = item["condition_id"],
        value = item["value"]
    ) for item in sell_rules]
    # Add user to database
    db.add_all(sell_rules_strategy)
    db.commit()

    backtest_keys = [item for item in form_data.keys() if 'backtest_' in item]
    filter_keys = [item for item in form_data.keys() if 'filter_' in item]
    target_keys = [item for item in form_data.keys() if 'target_' in item]

    parameters = {"backtest": {},"filter":{},"target":{}}
    for item in backtest_keys:
        parameters["backtest"][item] = form_data[item]
    for item in filter_keys:
        parameters["filter"][item] = form_data[item]
    for item in target_keys:
        parameters["target"][item] = form_data[item]

    #strategy.strategy_parameters = str(parameters)
    db.query(Strategy).filter(Strategy.id == strategy.id).update(
        {Strategy.strategy_parameters:str(parameters),
        Strategy.name :form_data["name"],
        Strategy.description :form_data["description"],
        Strategy.buy_strategy_id: aux_buy_strategy_id,
        Strategy.user_id: user.id,
        Strategy.json_parameters: str(form_data)
        })
    db.commit()
    print(parameters)

    return RedirectResponse("/settings",status_code=303)

# ################# TRAINER #####################

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

@router.post("/trainer/new", response_class=RedirectResponse)
async def create_trainer(
        request: Request, user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
    form_data = await request.form()
    form_data = tuples_to_dict(form_data.multi_items())
    print(form_data)
    
    training_detail = TrainingDetail(
        name = form_data['name'],
        description = form_data['description'],
        data_split = form_data['split_data'],
        user_id = user.id
    )
    db.add(training_detail)
    db.commit()
    db.refresh(training_detail)

    for item in form_data["algorithms[]"]:
        algo_params = [ val for val in form_data.keys() if item+"_" in val]
        aux_dict = {val.split('_', 1)[1]:form_data[val] for val in algo_params}
        algorithm_detail= AlgorithmDetail(
            training_detail_id = training_detail.id,
            algorithm_id = item,
            parameters = str(aux_dict)
        )
        db.add(algorithm_detail)
        db.commit()
    
    strategy_pairs = [TrainingMetric(
        training_detail_id = training_detail.id,
        model_metric_id = item
    ) for item in form_data["model_metrics[]"]]
    # Add user to database
    db.add_all(strategy_pairs)
    db.commit()

    training_details = db.query(TrainingDetail).order_by(TrainingDetail.created_at).all()
    print(training_details)

    return RedirectResponse("/settings",status_code=303)

@router.get("/trainer/edit/{id_train}", response_class=HTMLResponse)
async def edit_trainer(request: Request,id_train: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    model_metrics = db.query(ModelMetric).order_by(ModelMetric.name).all()
    algorithms = db.query(Algorithm).order_by(Algorithm.name).all()
    aux_algorithms = []
    for item in algorithms:
        aux_algorithms.append({"id":item.id, "name": item.name, "parameters": ast.literal_eval(item.parameters)})
    algorithms = aux_algorithms

    training_detail = db.query(TrainingDetail).get(id_train)
    training_algorithms = [item.algorithm_id for item in training_detail.algorithm_details]
    training_algorithms_parameters = {item.algorithm_id:ast.literal_eval(item.parameters) for item in training_detail.algorithm_details}
    training_metrics = [item.model_metric_id for item in training_detail.training_metrics]

    return templates.TemplateResponse(
        "home/trainer/edit.html",
        {"request": request, "current_user": user, "segment": "trainer", "training_algorithms_parameters": training_algorithms_parameters, "model_metrics": model_metrics, "training_detail": training_detail, "algorithms": algorithms, "training_metrics": training_metrics, "training_algorithms": training_algorithms}
    )

@router.post("/trainer/edit", response_class=HTMLResponse)
async def update_trainer(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):

    form_data = await request.form()
    form_data = tuples_to_dict(form_data.multi_items())
    print(form_data)

    db.query(TrainingDetail).filter(TrainingDetail.id == form_data["training_detail_id"]).update(
        {
            TrainingDetail.name :form_data["name"],
            TrainingDetail.description :form_data["description"],
            TrainingDetail.data_split: form_data['split_data']
        })
    db.commit()

    training_detail = db.query(TrainingDetail).get(form_data["training_detail_id"])
    for item in training_detail.algorithm_details:
        db.delete(item)
        db.commit()
    for item in form_data["algorithms[]"]:
        algo_params = [ val for val in form_data.keys() if item+"_" in val]
        aux_dict = {val.split('_', 1)[1]:form_data[val] for val in algo_params}
        algorithm_detail= AlgorithmDetail(
            training_detail_id = training_detail.id,
            algorithm_id = item,
            parameters = str(aux_dict)
        )
        db.add(algorithm_detail)
        db.commit()
    
    for item in training_detail.training_metrics:
        db.delete(item)
        db.commit()
    training_metrics = [TrainingMetric(
        training_detail_id = training_detail.id,
        model_metric_id = item
    ) for item in form_data["model_metrics[]"]]
    # Add user to database
    db.add_all(training_metrics)
    db.commit()

    training_details = db.query(TrainingDetail).order_by(TrainingDetail.created_at).all()
    print(training_details)

    return RedirectResponse("/settings",status_code=303)

# ################# TRADING ##########################

@router.get("/trading/new", response_class=HTMLResponse)
async def new_trading(request: Request, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        "home/trading/new.html",
        {"request": request,
            "current_user": user, 
            "segment": "trading"
        }
    )

@router.post("/trading/new", response_class=RedirectResponse)
async def create_trading(
        request: Request, user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
    ):
    form_data = await request.form()
    form_data = tuples_to_dict(form_data.multi_items())
    print(form_data)
    
    trading_setup = TradingSetup(
        name = form_data["name"],
        description = form_data["description"],
        exchange = form_data["exchange"],
        exchange_comission = form_data["commission"],
        capital = form_data["capital"],
        currency_base = form_data["currency"],
        api_key = form_data["api_key"],
        api_secret = form_data["secret_key"],
        user_id = user.id
    )
    db.add(trading_setup)
    db.commit()

    return RedirectResponse("/settings",status_code=303)

@router.get("/trading/edit/{id_trading}", response_class=HTMLResponse)
async def edit_trading(request: Request,id_trading: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    trading_setup = db.query(TradingSetup).filter(TradingSetup.id==id_trading).first()
    
    return templates.TemplateResponse(
        "home/trading/edit.html",
        {"request": request,
            "current_user": user, 
            "segment": "trading",
            "trading_setup":trading_setup
        }
    )

@router.post("/trading/edit", response_class=RedirectResponse)
async def update_trading(
        request: Request,
        db: Session = Depends(get_db)
    ):
    form_data = await request.form()
    form_data = tuples_to_dict(form_data.multi_items())
    print(form_data)
    
    db.query(TradingSetup).filter(TradingSetup.id == form_data["trading_setup_id"]).update(
        {
            TradingSetup.name:form_data["name"],
            TradingSetup.description:form_data["description"],
            TradingSetup.exchange:form_data["exchange"],
            TradingSetup.exchange_comission:form_data["commission"],
            TradingSetup.capital:form_data["capital"],
            TradingSetup.currency_base:form_data["currency"],
            TradingSetup.api_key:form_data["api_key"],
            TradingSetup.api_secret:form_data["secret_key"]
        })
    db.commit()

    return RedirectResponse("/settings",status_code=303)



# ##################### EXTRA ###########################
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

@router.get("/buystrategy/get/{id}")
async def get_buystrategy(id: int, db: Session = Depends(get_db)):
    bs = db.query(BuyStrategy).filter(BuyStrategy.id == id).first()
    return JSONResponse(content={"id":bs.id,"parameters":ast.literal_eval(bs.parameters),"name":bs.name,"description":bs.description},status_code=200)

@router.get("/strategy/{id}/parameters/{buy_strat_id}")
async def get_strategy_parameters(id: int,buy_strat_id: int, db: Session = Depends(get_db)):
    bs = db.query(Strategy).filter(Strategy.id == id, Strategy.buy_strategy_id == buy_strat_id).first()
    return JSONResponse(content={"id":bs.id,"parameters":ast.literal_eval(bs.strategy_parameters)},status_code=200)