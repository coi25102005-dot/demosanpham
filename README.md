# EduPredict
Dùng Python 3.11 hoặc 3.12 trên Windows. Mở terminal tại thư mục chứa app.py và requirements.txt.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Nếu chỉ có Python 3.11, thay -3.12 bằng -3.11. Trình duyệt mở localhost:8501.
Tải dữ liệu tại https://archive.ics.uci.edu/dataset/320/student+performance, giải nén và đặt student-mat.csv cạnh app.py hoặc tải lên thanh bên. Nếu chưa có, ứng dụng dùng dữ liệu mô phỏng có cảnh báo. Không dùng số liệu mô phỏng làm kết quả nghiên cứu.
CSV hàng loạt có mẫu tải trong ứng dụng. Lịch sử lưu history.sqlite. Đây là demo cục bộ, không có đăng nhập hay phân quyền.
