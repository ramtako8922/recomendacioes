from typing import Optional
from pydantic import BaseModel


class MsgPayload(BaseModel):
    msg_id: Optional[int]
    msg_name: str

class UserInput(BaseModel):
    Month: int
    Accommodation_type: int
    Transportation_type: int
    Male: int
    Female: int
