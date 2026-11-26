import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import sqlite3, io, zipfile

st.set_page_config(page_title='EduPredict | Dự đoán điểm cuối kỳ', page_icon='🎓', layout='wide')
FEATURES=['G1','G2','studytime','absences','failures','Medu','Fedu','schoolsup','famsup']
LABELS=['Điểm kỳ đánh giá 1 (0–20)','Điểm kỳ đánh giá 2 (0–20)','Thời gian học (1–4)','Số buổi vắng','Số lần học trượt','Học vấn mẹ (0–4)','Học vấn bố (0–4)','Hỗ trợ từ trường','Hỗ trợ gia đình']
BOUNDS=[(0,20),(0,20),(1,4),(0,93),(0,3),(0,4),(0,4),(0,1),(0,1)]
DB=Path(__file__).with_name('history.sqlite')
def read_csv(raw):
    return pd.read_csv(io.BytesIO(raw),sep=None,engine='python')
def prepare(df, target=False):
    cols=FEATURES+(['G3'] if target else [])
    missing=set(cols)-set(df.columns)
    if missing: raise ValueError('Thiếu cột: '+', '.join(sorted(missing)))
    x=df[cols].copy()
    for c in ['schoolsup','famsup']: x[c]=x[c].replace({'yes':1,'no':0,'Yes':1,'No':0})
    for c,(lo,hi) in zip(cols,BOUNDS+([(0,20)] if target else [])):
        x[c]=pd.to_numeric(x[c],errors='raise')
        if x[c].isna().any() or not x[c].between(lo,hi).all(): raise ValueError(f'{c}: cần giá trị từ {lo} đến {hi}, không được bỏ trống')
    return x
@st.cache_data
def real_data():
    from urllib.request import urlopen
    url='https://archive.ics.uci.edu/static/public/320/student+performance.zip'
    with urlopen(url,timeout=30) as response:
        z=zipfile.ZipFile(io.BytesIO(response.read()))
    for name in z.namelist():
        if name.endswith('student-mat.csv'):
            return read_csv(z.read(name))
        if name.endswith('student.zip'):
            inner=zipfile.ZipFile(io.BytesIO(z.read(name)))
            return read_csv(inner.read('student-mat.csv'))
    raise ValueError('Không tìm thấy student-mat.csv trong dữ liệu UCI')
@st.cache_resource
def train(d):
    X=d[FEATURES]; y=d.G3; a,b,c,e=train_test_split(X,y,test_size=.2,random_state=42)
    models={'Linear Regression':LinearRegression(),'Ridge':make_pipeline(StandardScaler(),Ridge()),'Random Forest':RandomForestRegressor(n_estimators=150,random_state=42),'Gradient Boosting':GradientBoostingRegressor(random_state=42),'SVR':make_pipeline(StandardScaler(),SVR())}
    rows=[]; predictions={}
    for name,m in models.items():
        m.fit(a,c); p=np.clip(m.predict(b),0,20); predictions[name]=pd.DataFrame({'Thực tế':e.to_numpy(),'Dự đoán':p}); rows.append({'Mô hình':name,'MAE':mean_absolute_error(e,p),'RMSE':mean_squared_error(e,p)**.5,'R²':r2_score(e,p)})
    return models,pd.DataFrame(rows).sort_values('MAE'),predictions

def save(rows):
    st.session_state.prediction_history=pd.concat([history(),rows],ignore_index=True)
def history():
    return st.session_state.get('prediction_history',pd.DataFrame()).copy()
def download(df,name): st.download_button('⬇ Tải CSV',df.to_csv(index=False).encode('utf-8-sig'),name,'text/csv')
def result(x,model,name,source):
    out=x.copy(); out['Điểm dự đoán /20']=np.round(np.clip(model.predict(x[FEATURES]),0,20),2); out['Điểm dự đoán /10']=out['Điểm dự đoán /20']/2
    out['Cảnh báo']=np.where(out['Điểm dự đoán /10']<5,'Nguy cơ dưới 5','Không dưới 5'); out['Mô hình']=name; out['Nguồn']=source; out['Thời gian']=pd.Timestamp.now().isoformat(); return out
st.sidebar.title('🎓 EduPredict')
page=st.sidebar.radio('Chức năng',['Tổng quan','Dự đoán cá nhân','Dự đoán hàng loạt','Quản lý dữ liệu','Phân tích dữ liệu','So sánh mô hình','Cảnh báo học tập','Lịch sử dự đoán','Giới thiệu'])
file=st.sidebar.file_uploader('Dữ liệu huấn luyện student-mat.csv',type=['csv'])
try:
    if file: d=prepare(read_csv(file.getvalue()),True); source='CSV người dùng'
    elif Path(__file__).with_name('student-mat.csv').exists(): d=prepare(read_csv(Path(__file__).with_name('student-mat.csv').read_bytes()),True); source='student-mat.csv cục bộ'
    else: d=prepare(real_data(),True); source='UCI Student Performance (tải tự động)'
    if len(d)<30: raise ValueError('Cần ít nhất 30 bản ghi để chạy demo')
except Exception as ex: st.error(str(ex)); st.info('Tải student-mat.csv từ UCI rồi đưa vào thanh bên hoặc đặt cạnh app.py. Không dùng dữ liệu giả.'); st.stop()
models,scores,preds=train(d); name=st.sidebar.selectbox('Mô hình dự đoán',scores['Mô hình'].tolist()); model=models[name]
st.title(page); st.caption('Dự đoán điểm thi cuối kỳ • Môn Toán • G3')
st.caption('Nguồn: '+source)
if page=='Tổng quan':
    a,b,c,e=st.columns(4); a.metric('Số bản ghi',len(d)); b.metric('Điểm G3 trung bình /10',round(d.G3.mean()/2,2)); c.metric('Tỷ lệ G3 dưới 5',f'{(d.G3<10).mean():.1%}'); e.metric('MAE tốt nhất /20',f'{scores.MAE.min():.2f}')
    st.subheader('Phân bố điểm cuối kỳ'); st.bar_chart(d.G3.value_counts().sort_index()); st.subheader('Kết quả mô hình'); st.dataframe(scores,hide_index=True)
elif page=='Dự đoán cá nhân':
    with st.form('individual'):
        values={}; cols=st.columns(3)
        for i,(f,label,(lo,hi)) in enumerate(zip(FEATURES,LABELS,BOUNDS)):
            with cols[i%3]:
                if f in ['schoolsup','famsup']: values[f]=int(st.selectbox(label,['Không','Có'])=='Có')
                else: values[f]=st.number_input(label,min_value=lo,max_value=hi,value=12 if f in ['G1','G2'] else (2 if f=='studytime' else lo))
        submit=st.form_submit_button('Dự đoán điểm',type='primary')
    if submit:
        x=pd.DataFrame([values]); out=result(x,model,name,source); save(out); st.session_state.last=out
    if 'last' in st.session_state:
        out=st.session_state.last; r=out.iloc[0]; a,b=st.columns(2); a.metric('Điểm dự đoán /20',r['Điểm dự đoán /20']); b.metric('Điểm quy đổi /10',f"{r['Điểm dự đoán /10']:.2f}"); st.write('Cảnh báo:',r['Cảnh báo']); st.caption('Mô hình: '+r['Mô hình']+' | '+r['Nguồn']); st.subheader('Gợi ý học tập')
        if r.G2<r.G1: st.write('• Ôn lại kiến thức ở kỳ đánh giá gần nhất.')
        if r.studytime<=2: st.write('• Lập lịch tự học đều đặn.')
        if r.absences>5: st.write('• Hạn chế nghỉ học và bổ sung bài đã bỏ lỡ.')
        if r.failures>0: st.write('• Trao đổi với giáo viên về phần kiến thức còn yếu.')
        st.write('• Theo dõi tiến độ qua các bài kiểm tra tiếp theo.'); download(out,'du-doan.csv')
elif page=='Dự đoán hàng loạt':
    st.write('Tải CSV có 9 cột đầu vào. Không cần cột G3.'); st.download_button('Tải CSV mẫu',d[FEATURES].head(5).to_csv(index=False),'mau.csv'); f=st.file_uploader('Danh sách học sinh',type=['csv'])
    if f and st.button('Chạy dự đoán'):
        try:
            out=result(prepare(read_csv(f.getvalue())),model,name,source); save(out); st.session_state.batch=out
        except Exception as ex: st.error(str(ex))
    if 'batch' in st.session_state: st.dataframe(st.session_state.batch); download(st.session_state.batch,'du-doan-hang-loat.csv')
elif page=='Quản lý dữ liệu':
    lo,hi=st.slider('Lọc G3 /20',0,20,(0,20)); q=st.text_input('Tìm theo số thứ tự bản ghi'); view=d[d.G3.between(lo,hi)]
    if q: view=view[view.index.astype(str).str.contains(q,regex=False)]
    st.dataframe(view); download(view,'du-lieu.csv')
elif page=='Phân tích dữ liệu':
    st.subheader('Tương quan với G3'); st.bar_chart(d.corr()['G3'].drop('G3')); st.subheader('G2 và G3'); st.scatter_chart(d,x='G2',y='G3'); st.subheader('Điểm trung bình theo thời gian học'); st.bar_chart(d.groupby('studytime').G3.mean()); st.dataframe(d.describe())
elif page=='So sánh mô hình':
    st.caption('Tập kiểm tra 20%, random_state=42; chỉ số tính trên thang 20.'); st.dataframe(scores,hide_index=True); st.bar_chart(scores.set_index('Mô hình')[['MAE','RMSE']]); p=preds[name].copy(); p['Sai số']=p['Thực tế']-p['Dự đoán']; st.scatter_chart(p,x='Thực tế',y='Dự đoán'); st.bar_chart(p['Sai số'].round().value_counts().sort_index()); download(scores,'metrics.csv')
elif page=='Cảnh báo học tập':
    h=history(); threshold=st.slider('Ngưỡng cảnh báo /10',0.,10.,5.,.5)
    if h.empty: st.info('Chưa có dự đoán. Hãy dự đoán cá nhân hoặc hàng loạt trước.')
    else:
        risk=h[h['Điểm dự đoán /10']<threshold]; st.metric('Số lượt dự đoán dưới ngưỡng',len(risk)); st.dataframe(risk); download(risk,'canh-bao.csv')
elif page=='Lịch sử dự đoán':
    h=history(); st.dataframe(h); download(h,'lich-su.csv')
    if st.checkbox('Xác nhận xóa toàn bộ lịch sử') and st.button('Xóa lịch sử'):
        st.session_state.prediction_history=pd.DataFrame()
        st.rerun()
else:
    st.markdown('''### Mục tiêu\nDự đoán G3 môn Toán (điểm cuối năm trong UCI) từ thông tin học tập đã có trước thời điểm chốt điểm. Đây là bài toán hồi quy.\n### Dữ liệu\nUCI Student Performance: student-mat.csv, 395 bản ghi, 33 thuộc tính trong dữ liệu gốc. Demo chọn 9 thuộc tính đầu vào. G1/G2 là điểm các kỳ đánh giá trước.\n[Tải dữ liệu gốc UCI](https://archive.ics.uci.edu/dataset/320/student+performance)\n### Giới hạn\nĐiểm dự đoán không phải điểm chính thức. Dữ liệu UCI thu thập tại Bồ Đào Nha, chưa được kiểm định cho học sinh Việt Nam. Gợi ý là quy tắc tham khảo, không phải suy luận nhân quả. Mô hình cần G2 nên không dùng để dự đoán từ đầu năm. Lịch sử chỉ lưu trong phiên trình duyệt; không có xác thực người dùng.''')
