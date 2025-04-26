import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
BASE_URL = 'https://paper-api.alpaca.markets'
STOCK = 'AAPL'
TIMEFRAME = '1D'
WINDOW = 14

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def fetch_data(stock, timeframe, limit=100):
    barset = api.get_bars(stock, timeframe, limit=limit).df
    df = barset[barset.symbol == stock].copy()
    df['rsi'] = compute_rsi(df['close'], WINDOW)
    df['sma'] = df['close'].rolling(window=WINDOW).mean()
    df.dropna(inplace=True)
    return df

def compute_rsi(series, window):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def prepare_model_data(df):
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df[['rsi', 'sma']]
    y = df['target']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def predict_next_move(model, latest_data):
    features = latest_data[['rsi', 'sma']].iloc[-1:]
    prediction = model.predict(features)[0]
    return prediction

def make_trade(prediction):
    try:
        position = api.get_position(STOCK)
        has_position = int(position.qty) > 0
    except:
        has_position = False

    if prediction == 1 and not has_position:
        print("BUYING")
        api.submit_order(symbol=STOCK, qty=1, side='buy', type='market', time_in_force='gtc')
    elif prediction == 0 and has_position:
        print("SELLING")
        api.submit_order(symbol=STOCK, qty=1, side='sell', type='market', time_in_force='gtc')

if __name__ == '__main__':
    df = fetch_data(STOCK, TIMEFRAME)
    X_train, X_test, y_train, y_test = prepare_model_data(df)
    model = train_model(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {acc:.2f}")
    
    prediction = predict_next_move(model, df)
    print(f"Prediction for next move: {'BUY' if prediction == 1 else 'SELL'}")
    make_trade(prediction)
