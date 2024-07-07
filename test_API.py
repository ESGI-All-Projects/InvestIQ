from datetime import datetime

from API.trading import create_order, get_position, close_position
from API.current_stock_data import get_all_active_stocks
from API.account import get_account, get_account_portfolio_gain
from API.trading import get_all_positions_repartition

# response = get_all_active_stocks()
# print(response)

# response = create_order('AAPL', '50')
# print(response)

# response = close_position('AAPL')
# print(response)

# response = get_position('AAPL')
# print(response)

# get_account()

start_date = datetime.fromisoformat("2024-06-23 17:53:00")
#
gain, current_date = get_account_portfolio_gain(start_date)
#
print(gain)
print(current_date)

# res = get_all_positions_repartition()
# print(res)
