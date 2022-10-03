# -*- encoding: utf-8 -*-

# models.py

from symbol import parameters
from ..database import Base
from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String, DateTime, FetchedValue, Boolean, ForeignKey, Float, Date

class Algorithm(Base):

    __tablename__ = "algorithm"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    parameters = Column(String(2000))
    type = Column(String(200))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    training_details = relationship("TrainingDetail", back_populates="algorithm")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class AlgorithmDetail(Base):

    __tablename__ = "algorithm_detail"
    #normal columns
    id = Column(Integer, primary_key=True)
    training_detail_id = Column(Integer)
    algorithm_id = Column(Integer)
    name = Column(String(200))
    parameters = Column(String(3000))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    # Relation with parents Has one
    training_detail = relationship("TrainingDetail", back_populates="algorithm_details")
    algorithm = relationship("Algorithm", back_populates="algorithm_details")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())
class Backtest(Base):

    __tablename__ = "backtest"
    #normal columns
    id = Column(Integer, primary_key=True)
    #strategy_id = Column(Integer, ForeignKey("strategy.id"))
    train_id = Column(Integer, ForeignKey("train.id"))

    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    backtest_charts = relationship("BacktestChart", back_populates="backtest")
    backtest_metrics = relationship("BacktestMetric", back_populates="backtest")
    # Relation with parents Has one
    #training_detail = relationship("TrainingDetail", back_populates="backtests")
    #strategy = relationship("Strategy", back_populates="backtests")
    train = relationship("Train", back_populates="backtests")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class BacktestChart(Base):

    __tablename__ = "backtest_chart"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    path = Column(String(200))
    backtest_id = Column(Integer, ForeignKey("backtest.id"))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many

    # Relation with parents Has one
    backtest = relationship("Backtest", back_populates="backtest_charts")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class BacktestMetric(Base):
    __tablename__ = "backtest_metric"
    #normal columns
    id = Column(Integer, primary_key=True)
    value = Column(Float)
    description = Column(String(200))
    backtest_id = Column(Integer, ForeignKey("backtest.id"))
    portfolio_metric_id = Column(Integer, ForeignKey("portfolio_metric.id"))
    # Relation with childrens Has many

    # Relation with parents Has one
    backtest = relationship("Backtest", back_populates="backtest_metrics")
    portfolio_metric = relationship("PortfolioMetric", back_populates="backtest_metrics")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class BuyStrategy(Base):

    __tablename__ = "buy_strategy"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    parameters = Column(String(3000))
    description = Column(String(1000))
    active = Column(Boolean, default=True)
    
    # Relation with childrens Has many
    strategies = relationship("Strategy", back_populates="buy_strategy")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Condition(Base):
    __tablename__ = "condition"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(500))
    active = Column(Boolean, default=True)
    
    # Relation with childrens Has many

    # NO BIDIRECCIONAL

    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Indicator(Base):

    __tablename__ = "indicator"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(20))
    description = Column(String(200))
    params = Column(String(500))
    active = Column(Boolean, default=True)
    
    # Relation with childrens Has many
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class ModelMetric(Base):

    __tablename__ = "model_metric"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    training_metrics = relationship("TrainingMetric", back_populates="model_metric")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class PortfolioMetric(Base):

    __tablename__ = "portfolio_metric"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    strategy_metrics = relationship("StrategyMetric", back_populates="portfolio_metric")
    backtest_metrics = relationship("BacktestMetric", back_populates="portfolio_metric")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Pair(Base):

    __tablename__ = "pair"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(20))
    exchange = Column(String(100))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    strategy_pairs = relationship("StrategyPair", back_populates="pair")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Rule(Base):
    __tablename__ = "rule"
    #normal columns
    id = Column(Integer, primary_key=True)
    order_type = Column(Boolean, default=True) # True == Buy False == Sell
    strategy_id = Column(Integer, ForeignKey("strategy.id"))
    first_indicator_id = Column(Integer, ForeignKey("indicator.id"))
    second_indicator_id = Column(Integer, ForeignKey("indicator.id"))
    condition_id = Column(Integer, ForeignKey("condition.id"))
    # Relation with childrens Has many

    # Relation with parents Has one
    strategy = relationship("Strategy", back_populates="rules")
    first_indicator = relationship("Indicator", foreign_keys='Rule.first_indicator_id')
    second_indicator = relationship("Indicator", foreign_keys='Rule.second_indicator_id')
    condition = relationship("Condition")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

"""
class SellStrategy(Base):

    __tablename__ = "sell_strategy"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(1000))
    active = Column(Boolean, default=True)
    
    # Relation with childrens Has many
    strategies = relationship("Strategy", back_populates="sell_strategy")
    # Relation with parents Has one

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())
"""
class Strategy(Base):

    __tablename__ = "strategy"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(1000))
    strategy_parameters = Column(String(3000))
    buy_strategy_id = Column(Integer, ForeignKey("buy_strategy.id"))
    user_id = Column(Integer, ForeignKey("user.id"))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    rules = relationship("Rule", back_populates="strategy")
    strategy_pairs = relationship("StrategyPair", back_populates="strategy")
    training_settings = relationship("TrainSetup", back_populates="strategy")
    trades = relationship("Trade", back_populates="strategy")
    strategy_metrics = relationship("StrategyMetric", back_populates="strategy")
    # Relation with parents Has one
    buy_strategy = relationship("BuyStrategy", back_populates="strategies")
    user = relationship("User", back_populates="strategies")

    created_at = Column(DateTime, default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime, onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class StrategyMetric(Base):
    __tablename__ = "strategy_metric"
    #normal columns
    id = Column(Integer, primary_key=True)
    value = Column(Float)
    description = Column(String(200))
    strategy_id = Column(Integer, ForeignKey("strategy.id"))
    portfolio_metric_id = Column(Integer, ForeignKey("portfolio_metric.id"))
    # Relation with childrens Has many

    # Relation with parents Has one
    strategy = relationship("Strategy", back_populates="strategy_metrics")
    portfolio_metric = relationship("PortfolioMetric", back_populates="strategy_metrics")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class StrategyPair(Base):
    __tablename__ = "strategy_pair"
    #normal columns
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey("strategy.id"))
    pair_id = Column(Integer, ForeignKey("pair.id"))
    # Relation with childrens Has many

    # Relation with parents Has one
    strategy = relationship("Strategy", back_populates="strategy_pairs")
    pair = relationship("Pair", back_populates="strategy_pairs")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Trade(Base):
    __tablename__ = "trade"
    #normal columns
    id = Column(Integer, primary_key=True)
    trading_setup_id = Column(Integer, ForeignKey("trading_setup.id"))
    strategy_id = Column(Integer, ForeignKey("strategy.id"))
    quantity = Column(Float)
    buy_amount = Column(Float)
    sell_amount = Column(Float)
    entry_price = Column(Float)
    sell_price = Column(Float)
    stop_price = Column(Float)
    duration = Column(Integer) # cuantas velas demoró en cerrarse
    state = Column(Integer)  # Abierto 1, Cerrado 0, Stop -1
    # Relation with childrens Has many

    # Relation with parents Has one
    trading_setup = relationship("TradingSetup", back_populates="trades")
    strategy = relationship("Strategy", back_populates="trades")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class TradingSetup(Base):

    __tablename__ = "trading_setup"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(1000))
    active = Column(Boolean, default=True)
    exchange = Column(String(2000))
    exchange_comission = Column(Float(precision=4))
    capital = Column(Float(precision=8))
    currency_base = Column(String(10))
    currency_symbol = Column(String(5))
    api_key = Column(String(100))
    api_secret = Column(String(100))
        
    user_id = Column(Integer, ForeignKey("user.id"))

    # Relation with childrens Has many
    trades = relationship("Trade", back_populates="trading_setup")
    # Relation with parents Has one
    user = relationship("User", back_populates="trading_settings")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class Train(Base):

    __tablename__ = "train"
    #normal columns
    id = Column(Integer, primary_key=True)
    train_setup_id = Column(Integer, ForeignKey("train_setup.id"))
    scaler = Column(String(200))
    model = Column(String(200))
    algorithm_parameters = Column(String(3000))    
    strategy_parameters = Column(String(3000))    
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    backtests = relationship("Backtest", back_populates="train")
    # Relation with parents Has one
    train_setup = relationship("TrainSetup", back_populates="trainings")

    created_at = Column(DateTime, default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime, onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class TrainingDetail(Base):

    __tablename__ = "training_detail"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(1000))
    user_id = Column(Integer, ForeignKey("user.id"))
    #scaler = Column(String(200))
    #model = Column(String(200))
    active = Column(Boolean, default=True)

    # Relation with childrens Has many
    train_settings = relationship("TrainSetup", back_populates="training_detail")
    algorithm_details = relationship("AlgorithmDetail", back_populates="training_detail")
    training_metrics = relationship("TrainingMetric", back_populates="training_detail")
    # Relation with parents Has one
    user = relationship("User", back_populates="strategies")

    created_at = Column(DateTime, default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime, onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class TrainingMetric(Base):
    __tablename__ = "training_metric"
    #normal columns
    id = Column(Integer, primary_key=True)
    value = Column(Float)
    description = Column(String(200))
    training_detail_id = Column(Integer, ForeignKey("training_detail.id"))
    model_metric_id = Column(Integer, ForeignKey("model_metric.id"))
    # Relation with childrens Has many

    # Relation with parents Has one
    training_detail = relationship("TrainingDetail", back_populates="training_metrics")
    model_metric = relationship("ModelMetric", back_populates="training_metrics")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class TrainSetup(Base):

    __tablename__ = "train_setup"
    #normal columns
    id = Column(Integer, primary_key=True)
    name = Column(String(200))
    description = Column(String(1000))
    active = Column(Boolean, default=True)
    timeframe = Column(String(10))
    train_data_start = Column(Date)
    train_data_end = Column(Date)
    backtest_data_start = Column(Date)
    backtest_data_end = Column(Date)

    user_id = Column(Integer, ForeignKey("user.id"))
    trading_setup_id = Column(Integer, ForeignKey("trading_setup.id"))
    training_detail_id = Column(Integer, ForeignKey("training_detail.id"))
    strategy_id = Column(Integer, ForeignKey("strategy.id"))

    # Relation with childrens Has many
    trainings = relationship("Train", back_populates="train_setup")
    # Relation with parents Has one
    user = relationship("User", back_populates="training_settings")
    trading_setup = relationship("TradingSetup", back_populates="training_settings")
    training_detail = relationship("TrainingDetail", back_populates="training_settings")
    strategy = relationship("Strategy", back_populates="training_settings")

    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())

class User(Base):

    __tablename__ = "user"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True)
    email = Column(String(128), unique=True)
    password = Column(String(64))
    strategies = relationship("Strategy", back_populates="user")
    training_details = relationship("TrainingDetail", back_populates="user")
    training_settings = relationship("TrainSetup", back_populates="user")
    trading_settings = relationship("TradingSetup", back_populates="user")
    created_at = Column(DateTime(), default=datetime.now(), server_default=FetchedValue())
    updated_at = Column(DateTime(), onupdate=datetime.now(), server_default=FetchedValue(), server_onupdate=FetchedValue())