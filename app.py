from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI

from typing import Optional

app = FastAPI()

# Define what the app does
@app.get("/greet")
async def index(fname: Optional[str] = None, lname: Optional[str] = None):

    if not fname and not lname:
        # If both first name and last name are missing, return an error
        response = { "status" : "error" }
    elif fname and not lname:
        # If first name is present but last name is missing
        response = { "data" : f"Hello, {fname} !" }
    elif not fname and lname:
        # If first name is missing but last name is present
        response = { "data" : f"Hello, Mr. {lname} !" }
    else :
        # if none of the above is true, then both names must be present
        response = { "data" : f"Is your name {fname} {lname} ?" }

    return jsonable_encoder(response)

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0')
