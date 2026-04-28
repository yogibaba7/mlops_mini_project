
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,computed_field,Field

from typing import List,Annotated

from preprocessing_utils import PreprocessText
from model_loading import LoadModel,Vectorization



class Schema(BaseModel):
    sentiment : Annotated[str,Field(...,description="Text of User")]

    @computed_field
    @property
    def Preprocessedsentiment(self)->str:
        return PreprocessText(self.sentiment)
    

data = "i'm yogesh chouhan i have lot of thing today i'm very happy and performing outstanding"

dataobj = Schema(sentiment = data)
sentiment = dataobj.Preprocessedsentiment
sentiment_array = Vectorization(sentiment)


# API 
app = FastAPI()

@app.get("/")
def Home():
    return {"Message":"This is API Of SentimentAnalysis Model"}

@app.post("/predict")
def Predict(UserText:Schema):
    text = UserText.Preprocessedsentiment
    text_array = Vectorization(text)
    model = LoadModel("my_model","Production")
    prediction = model.predict(text_array)
    print(prediction)
    return JSONResponse(content={"Prediction":int(prediction[0])},status_code=200)



    





