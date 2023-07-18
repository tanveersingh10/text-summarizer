from fastapi import FastAPI
import uvicorn 
import os
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textSummarizer.pipeline.prediction import PredictionPipeline


app = FastAPI()

@app.get("/", tags=["authentication"]) #authentication automatically generates API documentation
async def index():
    return RedirectResponse(url="/docs")


#method that will call main.py which runs the entire model training pipeline from data ingestion to evaluation
@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    


#takes in text and returns summary
@app.post("/predict")
async def predict_route(text):
    try:

        obj = PredictionPipeline()
        text = obj.predict(text)
        return text
    except Exception as e:
        raise e
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)