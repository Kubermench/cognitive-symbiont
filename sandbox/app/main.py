from fastapi import FastAPI
app=FastAPI()
@app.get("/healthz")
def healthz(): return {"status":"ok"}
@app.get("/echo/{x}")
def echo(x:str): return {"echo":x}
