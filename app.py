from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd, numpy as np, io, zipfile, urllib.request
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.inspection import permutation_importance
from functools import lru_cache
app=FastAPI(title='EduPredict'); ROOT=Path(__file__).parent
BASE=['studytime','absences','failures','Medu','Fedu','schoolsup','famsup']; FEATURES={'early':BASE,'full':BASE+['G1','G2']}
@lru_cache(maxsize=1)
def data():
 p=ROOT/'student-mat.csv'
 if not p.exists():
  raw=urllib.request.urlopen('https://archive.ics.uci.edu/static/public/320/student+performance.zip',timeout=30).read()
  z=zipfile.ZipFile(io.BytesIO(raw))
  if not any(n.endswith('student-mat.csv') for n in z.namelist()): z=zipfile.ZipFile(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith('.zip')))))
  p.write_bytes(z.read(next(n for n in z.namelist() if n.endswith('student-mat.csv'))))
 d=pd.read_csv(p,sep=';'); assert len(d)==395 and set(BASE+['G1','G2','G3']).issubset(d.columns), 'CSV không đúng cấu trúc UCI Toán'
 return d

def encode(d):
 d=d.copy()
 for c in ['schoolsup','famsup']: d[c]=d[c].replace({'yes':1,'no':0})
 return d
@lru_cache(maxsize=1)
def train():
 d=encode(data()); tr,te=train_test_split(np.arange(len(d)),test_size=.2,random_state=42); y=d.G3/2; out={}; rows=[]
 for mode,cols in FEATURES.items():
  X=d[cols]; models={'Linear Regression':LinearRegression(),'Ridge':Ridge(),'Random Forest':RandomForestRegressor(n_estimators=120,random_state=42),'Gradient Boosting':GradientBoostingRegressor(random_state=42),'SVR':SVR()}; fitted={}
  for name,est in models.items():
   m=make_pipeline(StandardScaler(),est); cv=-cross_val_score(m,X.iloc[tr],y.iloc[tr],cv=KFold(5,shuffle=True,random_state=42),scoring='neg_mean_absolute_error').mean(); m.fit(X.iloc[tr],y.iloc[tr]); pred=m.predict(X.iloc[te]); row=dict(mode=mode,model=name,cv=float(cv),mae=float(mean_absolute_error(y.iloc[te],pred)),rmse=float(mean_squared_error(y.iloc[te],pred)**.5),r2=float(r2_score(y.iloc[te],pred))); rows.append(row); fitted[name]=m
  best=min((r for r in rows if r['mode']==mode),key=lambda r:r['cv'])['model']; m=fitted[best]; pred=m.predict(X.iloc[te]); imp=permutation_importance(m,X.iloc[te],y.iloc[te],n_repeats=15,random_state=42,scoring='neg_mean_absolute_error'); out[mode]={'best':best,'models':fitted,'importance':dict(zip(cols,imp.importances_mean.tolist())),'actual':y.iloc[te].tolist(),'predicted':pred.tolist()}
 return out,rows
@app.get('/')
def home(): return FileResponse(ROOT/'index.html')
@app.get('/api/data')
def dataset():
 try:
  d=data(); e=encode(d); return {'count':len(d),'columns':len(d.columns),'mean':float(d.G3.mean()/2),'missing':int(d.isna().sum().sum()),'records':d.to_dict('records'),'correlation':e[BASE+['G1','G2','G3']].corr()['G3'].drop('G3').to_dict(),'distribution':d.G3.value_counts().sort_index().to_dict()}
 except Exception as e: raise HTTPException(503,str(e))
@app.get('/api/models')
def metrics():
 try:
  o,r=train(); return {'rows':r,'details':{k:{a:b for a,b in v.items() if a!='models'} for k,v in o.items()}}
 except Exception as e: raise HTTPException(503,str(e))
def predict_one(mode,values,model=None):
 cols=FEATURES[mode]; bounds={'studytime':(1,4),'absences':(0,93),'failures':(0,3),'Medu':(0,4),'Fedu':(0,4),'schoolsup':(0,1),'famsup':(0,1),'G1':(0,20),'G2':(0,20)}; x={}
 for c in cols:
  v=values[c]; v={'yes':1,'no':0}.get(v,v); v=float(v); lo,hi=bounds[c]
  if not np.isfinite(v) or not lo<=v<=hi or v%1: raise ValueError('Giá trị không hợp lệ: '+c)
  x[c]=v
 o,_=train(); info=o[mode]; name=model or info['best']; score=float(np.clip(info['models'][name].predict(pd.DataFrame([x],columns=cols))[0],0,10)); tips=[]
 if x['studytime']<=2: tips.append('Lập lịch tự học đều đặn, chia nhỏ nội dung ôn tập.')
 if x['absences']>5: tips.append('Bổ sung bài đã bỏ lỡ và trao đổi với giáo viên.')
 if mode=='full' and x['G2']<x['G1']: tips.append('Xem lại nội dung kỳ đánh giá gần nhất.')
 if x['failures']>0: tips.append('Ưu tiên củng cố kiến thức nền tảng.')
 tips.append('Làm bài kiểm tra thử và theo dõi tiến độ hằng tuần.')
 return {'score':round(score,2),'model':name,'tips':tips,'risk':score<5}
@app.post('/api/predict')
def predict(body:dict):
 try:return predict_one(body['mode'],body['values'],body.get('model'))
 except Exception as e:raise HTTPException(400,str(e))
@app.post('/api/batch/{mode}')
async def batch(mode:str,file:UploadFile=File(...)):
 try:
  raw=await file.read(2000001)
  if len(raw)>2000000:raise ValueError('File quá lớn')
  d=pd.read_csv(io.BytesIO(raw),sep=None,engine='python');
  if len(d)>1000:raise ValueError('Tối đa 1000 dòng')
  return {'records':[{'id':i+1,**predict_one(mode,r.to_dict())} for i,r in d.iterrows()]}
 except Exception as e:raise HTTPException(400,str(e))
