# APP

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel,computed_field,Field

from typing import List,Annotated

from API.preprocessing_utils import PreprocessText
from API.model_loading import LoadModel, LoadVector



class Schema(BaseModel):
    sentiment : Annotated[str,Field(...,description="Text of User")]

    @computed_field
    @property
    def Preprocessedsentiment(self)->str:
        return PreprocessText(self.sentiment)

# API 
app = FastAPI()
model = LoadModel("my_model","Production")
vector = LoadVector()


data = "i'm yogesh chouhan i have lot of thing today i'm very happy and performing outstanding"

dataobj = Schema(sentiment = data)
# sentiment = dataobj.Preprocessedsentiment
# sentiment_array = vector.transform(sentiment)




@app.get("/")
def Home():
    return {"Message":"This is API Of SentimentAnalysis Model"}

@app.post("/predict")
def Predict(UserText:Schema):
    text = UserText.Preprocessedsentiment
    text_array = vector.transform([text])
    prediction = model.predict(text_array)
    print(prediction)
    return JSONResponse(content={"Prediction":int(prediction[0])},status_code=200)



    





