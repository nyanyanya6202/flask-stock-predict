"""Flaskで株価を分析し、30期間先の終値と騰落方向を予測する。"""
import os
import re

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4096
START_DATE = "2023-01-05"
FEATURES = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


# 1. 株価を取得し、移動平均と変動率を計算する
# --------------------------------------------------
def download_prices(ticker, start=START_DATE):
    df = yf.download(
        ticker, start=start, auto_adjust=False,
        multi_level_index=False, progress=False, timeout=20,
    )
    if df.empty:
        raise ValueError("株価データが見つかりません。銘柄コードを確認してください。")

    if isinstance(df.columns, pd.MultiIndex):
        level = 0 if "Close" in df.columns.get_level_values(0) else -1
        df.columns = df.columns.get_level_values(level)

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    df["Adj Close"] = df["Adj Close"].fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0)
    if len(df) < 2:
        raise ValueError("株価データが少なすぎます。")

    df["SMA14"] = df["Close"].rolling(14).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["change_pct"] = (
        (df["Close"] - df["Open"])
        / df["Open"].replace(0, float("nan")) * 100
    )
    return df


# 2. 線形回帰で予測し、上昇・非上昇の正解率を求める
# --------------------------------------------------
def forecast_next_periods(prices, horizon=30):
    if horizon < 1 or len(prices) < max(150, horizon * 5):
        raise ValueError("予測には十分な株価データが必要です（通常150行以上）。")

    df = prices.copy()
    df["label"] = df["Close"].shift(-horizon)
    known = df.iloc[:-horizon].dropna(subset=["label", *FEATURES])
    future_x = df.iloc[-horizon:][FEATURES]

    # 古い80%で学習、新しい20%で検証。境界の30行は学習から除く
    train, test = train_test_split(known, test_size=0.2, shuffle=False)
    train = train.iloc[:-horizon]
    if len(train) < 40 or len(test) < 10 or future_x.isna().any().any():
        raise ValueError("学習または予測に使用できるデータが不足しています。")

    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(train[FEATURES], train["label"])
    test_pred = model.predict(test[FEATURES])

    # 30期間先の価格が、予測時点の終値より高いかを比較する
    actual_up = test["label"] > test["Close"]
    predicted_up = test_pred > test["Close"].to_numpy()
    accuracy = accuracy_score(actual_up, predicted_up)

    # 正解が分かる全データで学習し直し、将来30期間を予測する
    model.fit(known[FEATURES], known["label"])
    values = model.predict(future_x)

    weekends = (prices.index[-90:].dayofweek >= 5).sum() >= 4
    dates = pd.date_range(
        start=prices.index[-1] + pd.Timedelta(days=1),
        periods=horizon, freq="D" if weekends else "B",
    )
    future = pd.DataFrame({"Predict": values}, index=dates)
    return future, accuracy, weekends


# 3. Plotlyでグラフを作成する
# --------------------------------------------------
def chart_html(fig, height=350, title="価格（銘柄の取引通貨）"):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=40, r=20, t=35, b=40),
    )
    fig.update_yaxes(title_text=title)
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def make_charts(prices, future=None):
    fig = go.Figure()
    fig.add_scatter(x=prices.index, y=prices["Close"], name="終値", mode="lines")
    fig.add_scatter(x=prices.index, y=prices["SMA14"], name="14期間移動平均", mode="lines")
    charts = {"closing": chart_html(fig)}

    recent = prices.tail(100)
    fig = go.Figure()
    fig.add_scatter(x=recent.index, y=recent["change_pct"], name="当日の変動率", mode="lines")
    fig.add_hline(y=0, line_dash="dash")
    charts["change"] = chart_html(fig, 320, "変動率（%）")

    recent = prices.tail(180)
    fig = go.Figure(data=[go.Candlestick(
        x=recent.index, open=recent["Open"], high=recent["High"],
        low=recent["Low"], close=recent["Close"],
    )])
    fig.update_xaxes(rangeslider_visible=False)
    charts["candle"] = chart_html(fig, 420)

    if future is not None:
        fig = go.Figure()
        fig.add_scatter(x=recent.index, y=recent["Close"], name="実際の終値", mode="lines")
        fig.add_scatter(
            x=[prices.index[-1], *future.index],
            y=[prices["Close"].iloc[-1], *future["Predict"].tolist()],
            name="予測終値", mode="lines",
        )
        charts["prediction"] = chart_html(fig, 420)
    return charts


# 4. Flaskの画面に株価・予測結果を渡す
# --------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.route("/", methods=["GET", "POST"])
def index():
    page = dict(ticker="AAPL", error=None, has_data=False,
                summary={}, charts={}, table_rows=[], prediction=None)
    status = 200

    if request.method == "POST":
        ticker = request.form.get("ticker", "").strip().upper()
        action = request.form.get("action", "analyze")
        page["ticker"] = ticker

        try:
            if not re.fullmatch(r"[A-Z0-9^=._-]{1,18}", ticker):
                raise ValueError("銘柄コードを確認してください。")
            if action not in ("analyze", "predict"):
                raise ValueError("不正な操作です。")

            prices = download_prices(ticker)
            last, prev = prices.iloc[-1], prices.iloc[-2]
            change = (last["Close"] / prev["Close"] - 1) * 100 if prev["Close"] else None
            page["summary"] = {
                "last_date": prices.index[-1].strftime("%Y-%m-%d"),
                "close": f"{last['Close']:,.2f}",
                "previous_change_pct": f"{change:+.2f}%" if change is not None else "—",
                "sma14": f"{last['SMA14']:,.2f}" if len(prices) >= 14 else "—",
                "data_count": f"{len(prices):,}",
            }

            for day, row in prices.tail(20).iloc[::-1].iterrows():
                page["table_rows"].append({
                    "date": day.strftime("%Y-%m-%d"),
                    "open": f"{row['Open']:,.2f}", "high": f"{row['High']:,.2f}",
                    "low": f"{row['Low']:,.2f}", "close": f"{row['Close']:,.2f}",
                    "volume": f"{int(row['Volume']):,}",
                    "change": f"{row['change_pct']:+.2f}%",
                    "positive": bool(row["change_pct"] >= 0),
                })

            future = None
            if action == "predict":
                future, accuracy, weekends = forecast_next_periods(prices)
                page["prediction"] = {
                    "accuracy": f"{accuracy:.2%}",
                    "direction": "上昇／非上昇",
                    "calendar": "暦日（週末を含む）" if weekends else "平日換算（祝日は未考慮）",
                    "rows": [
                        {"date": day.strftime("%Y-%m-%d"), "price": f"{row['Predict']:,.2f}"}
                        for day, row in future.iterrows()
                    ],
                }

            page["charts"] = make_charts(prices, future)
            page["has_data"] = True

        except ValueError as exc:
            page["error"] = str(exc)
            status = 400
        except Exception:
            app.logger.exception("株価取得または予測中のエラー: %s", ticker)
            page["error"] = "処理に失敗しました。銘柄コード・通信状態を確認してください。"
            status = 500

    return render_template("index.html", **page), status


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", "5000")), debug=False)
