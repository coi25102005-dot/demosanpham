# EduPredict
Python 3.12 recommended. Open terminal in this folder.

```
py -3.12 -m pip install -r requirements.txt
py -3.12 -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000. Data automatically downloads from UCI; alternatively extract student-mat.csv from https://archive.ics.uci.edu/dataset/320/student+performance and place beside app.py. No synthetic fallback. G3 is final year grade, converted to /10. Early scenario excludes G1/G2 but absences may span the year. Not validated in Vietnam. History is browser session only. No personal data needed.

Deploy on Render as Python web service: build `pip install -r requirements.txt`, start `uvicorn app:app --host 0.0.0.0 --port $PORT`. Upload all files to GitHub first. Not a Streamlit app.
