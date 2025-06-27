# === Global State ===
cash = 100000.0  # Starting capital
PnL = 0.0
stocks = []  # Each entry: {"buy_price": float, "quantity": int}
options = []  # Each entry: {"strike": float, "expiry": datetime, "option_type": str, "quantity": int, "entry_price": float, "direction": "long" or "short"}
trade_log = []

import matplotlib.pyplot as plt
import pandas as pd

# === Strategy Function Stub ===
def strategy(timestamp, open_price, high_price, low_price, close_price, volume, options_chain, portfolio):
    """
    Strategy decides what to do at this timestep.
    Returns a list of action dictionaries.
    """
    actions = []

    # Example logic (placeholder):
    if close_price < open_price:
        # Buy stock if it's falling (example mean reversion idea)
        actions.append({"asset_type": "stock", "action": "buy", "quantity": 10})
    else:
        # Sell a put if it's rising (example volatility play)
        for opt in options_chain:
            if opt["option_type"] == "put" and opt["strike"] < close_price:
                actions.append({
                    "asset_type": "option",
                    "action": "sell",
                    "option_type": "put",
                    "strike": opt["strike"],
                    "expiry": opt["expiry"],
                    "quantity": 1
                })
                break

    return actions

# === Backtest Execution ===
def run_backtest(market_data, options_data):
    global cash, PnL, stocks, options, trade_log
    portfolio_values = []
    timestamps = []

    for data_point in market_data:
        timestamp = data_point["timestamp"]
        open_price = data_point["open"]
        high_price = data_point["high"]
        low_price = data_point["low"]
        close_price = data_point["close"]
        volume = data_point["volume"]

        # Filter options chain for current timestamp
        options_chain = [opt for opt in options_data if opt["timestamp"] == timestamp]

        portfolio = {
            "stocks": stocks,
            "options": options,
            "cash": cash,
            "PnL": PnL
        }

        # === Step 1: Call Strategy ===
        actions = strategy(timestamp, open_price, high_price, low_price, close_price, volume, options_chain, portfolio)

        # === Step 2: Execute Actions ===
        for action in actions:
            if action["asset_type"] == "stock":
                price = close_price  # Assuming market order
                qty = action["quantity"]
                cost = qty * price

                if action["action"] == "buy" and cash >= cost:
                    cash -= cost
                    stocks.append({"buy_price": price, "quantity": qty})
                    trade_log.append((timestamp, "BUY STOCK", qty, price))

                elif action["action"] == "sell":
                    # Sell from oldest lot
                    remaining = qty
                    new_stocks = []
                    for lot in stocks:
                        if remaining <= 0:
                            new_stocks.append(lot)
                        elif lot["quantity"] <= remaining:
                            realized = lot["quantity"] * (price - lot["buy_price"])
                            PnL += realized
                            cash += lot["quantity"] * price
                            trade_log.append((timestamp, "SELL STOCK", lot["quantity"], price, realized))
                            remaining -= lot["quantity"]
                        else:
                            realized = remaining * (price - lot["buy_price"])
                            PnL += realized
                            cash += remaining * price
                            new_stocks.append({"buy_price": lot["buy_price"], "quantity": lot["quantity"] - remaining})
                            trade_log.append((timestamp, "SELL STOCK", remaining, price, realized))
                            remaining = 0
                    stocks = new_stocks

            elif action["asset_type"] == "option":
                for opt in options_chain:
                    if opt["strike"] == action["strike"] and opt["expiry"] == action["expiry"] and opt["option_type"] == action["option_type"]:
                        price = opt["price"]
                        qty = action["quantity"]
                        direction = "long" if action["action"] == "buy" else "short"
                        cost = qty * price * 100

                        if direction == "long" and cash >= cost:
                            cash -= cost
                            options.append({"strike": opt["strike"], "expiry": opt["expiry"], "option_type": opt["option_type"], "quantity": qty, "entry_price": price, "direction": direction})
                            trade_log.append((timestamp, "BUY OPTION", opt["option_type"], qty, price))

                        elif direction == "short":
                            cash += cost  # Premium received
                            options.append({"strike": opt["strike"], "expiry": opt["expiry"], "option_type": opt["option_type"], "quantity": qty, "entry_price": price, "direction": direction})
                            trade_log.append((timestamp, "SELL OPTION", opt["option_type"], qty, price))

        # === Step 3: Handle Expiries ===
        new_options = []
        for opt in options:
            if opt["expiry"] == timestamp:
                intrinsic = 0
                if opt["option_type"] == "call":
                    intrinsic = max(0, close_price - opt["strike"])
                elif opt["option_type"] == "put":
                    intrinsic = max(0, opt["strike"] - close_price)
                payout = intrinsic * opt["quantity"] * 100
                if opt["direction"] == "long":
                    PnL += payout - (opt["entry_price"] * opt["quantity"] * 100)
                    trade_log.append((timestamp, "EXPIRE LONG OPTION", payout))
                else:
                    PnL += (opt["entry_price"] * opt["quantity"] * 100) - payout
                    trade_log.append((timestamp, "EXPIRE SHORT OPTION", -payout))
            else:
                new_options.append(opt)
        options = new_options

        # === Step 4: Track Portfolio Value ===
        portfolio_value = cash + sum(s["quantity"] * close_price for s in stocks)
        for opt in options:
            # Estimate option market value using current price if available
            for chain_opt in options_chain:
                if chain_opt["strike"] == opt["strike"] and chain_opt["option_type"] == opt["option_type"] and chain_opt["expiry"] == opt["expiry"]:
                    if opt["direction"] == "long":
                        portfolio_value += chain_opt["price"] * opt["quantity"] * 100
                    else:
                        portfolio_value -= chain_opt["price"] * opt["quantity"] * 100
        portfolio_values.append(portfolio_value)
        timestamps.append(timestamp)

    # === Final Portfolio Output ===
    result = {
        "final_cash": cash,
        "final_PnL": PnL,
        "stocks": stocks,
        "options": options,
        "trade_log": trade_log
    }

    # === Plotting ===
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, portfolio_values, label="Portfolio Value")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    plt.title("Backtest Portfolio Value Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return result
