from pydantic import BaseModel
from typing import List
# Pydantic BaseModel
# Order class model for request body
class Order(BaseModel):
    customer_name: str
    order_quantity: int

class Data(BaseModel):
    acc_x: List[float]
    acc_y: List[float]
    acc_z: List[float]