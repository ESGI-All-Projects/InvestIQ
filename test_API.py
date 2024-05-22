from src.API.trading import create_order, get_position
from src.API.current_stock_data import get_all_active_stocks

response = get_all_active_stocks()
print(response)

# response = create_order('AAPL', '1000')
# print(response)

# response = get_position('AAPL')
# print(response)