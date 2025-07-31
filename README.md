# 📝 Nhận dạng chữ viết tay tiếng Việt (Vietnamese Handwriting Recognition)

Dự án này nhằm xây dựng một hệ thống nhận dạng chữ viết tay tiếng Việt sử dụng Deep Learning. Mục tiêu là chuyển đổi hình ảnh dòng chữ viết tay thành văn bản số một cách chính xác và hiệu quả.

## 🚀 Tính năng chính

- Nhận dạng chữ viết tay dòng (line-level)
- Hỗ trợ tiếng Việt có dấu
- Huấn luyện mô hình từ đầu với PyTorch
- Tích hợp CRNN (Convolutional Recurrent Neural Network)
- Tự động tiền xử lý ảnh viết tay (crop, resize, padding,...)

## Dataset
- Tập dữ liệu được tải [ở đây](https://drive.google.com/file/d/1-hAGX91o45NA4nv1XUYw5pMw4jMmhsh5/view)
- Thư mục chứa dữ liệu chữ viết tay tiếng Việt là "InkData_line_processed"
- Cấu trúc thư mục dữ liệu như sau:
    ```
  data
  ├── lines
  │   ├── 13656456.png  
  │   ├── 13656456.txt
  │   ├── 789465116.png
  │   ├── 789465116.txt
  │   └── ...
  └── splits
      ├── train.txt
        └── val.txt
  ```
## 🔧 Cài đặt

```bash
git clone https://github.com/Quan0310/ocr_vi.git
cd ocr_vi
pip install -r requirements.txt

python train.py --epochs 50 --batch_size 16

```

## 📊 Biểu đồ hàm mất mát 
<img width="654" height="360" alt="image" src="https://github.com/user-attachments/assets/ec23956f-4732-423a-aa63-a7382a31f9cb" />

## 🧪 Thử nghiệm
```bash
# nếu ảnh chưa xử lý đen trắng
python test.py --image_path [image_path]
```
```bash
# nếu ảnh đã xử lý đen trắng
python test.py --i --image_path [image_path]
```
### Thử nghiệm với ảnh 1
![test1](https://github.com/user-attachments/assets/7c4ef906-a92a-467d-a02e-a2381e0921b5) 
### Kết quả
<img width="701" height="203" alt="image" src="https://github.com/user-attachments/assets/98f4b7ae-18f1-4cff-ad25-fb6097123d67" />

### Thử nghiệm với ảnh 2
![test2](https://github.com/user-attachments/assets/834fd0bb-9dd2-44dd-a046-dac069719573)
### Kết quả
<img width="574" height="236" alt="image" src="https://github.com/user-attachments/assets/dc30e679-72d3-4dd4-b7e1-f20f649e4117" />
