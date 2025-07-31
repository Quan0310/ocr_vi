# 📝 Nhận dạng chữ viết tay tiếng Việt (Vietnamese Handwriting Recognition)

Dự án này nhằm xây dựng một hệ thống nhận dạng chữ viết tay tiếng Việt sử dụng Deep Learning. Mục tiêu là chuyển đổi hình ảnh dòng chữ viết tay thành văn bản số một cách chính xác và hiệu quả.

## 🚀 Tính năng chính

- Nhận dạng chữ viết tay dòng (line-level)
- Hỗ trợ tiếng Việt có dấu
- Huấn luyện mô hình từ đầu với PyTorch
- Tích hợp CRNN (Convolutional Recurrent Neural Network)
- Tự động tiền xử lý ảnh viết tay (crop, resize, padding,...)

## 🔧 Cài đặt

```bash
git clone https://github.com/Quan0310/ocr_vi.git
cd ocr_vi
pip install -r requirements.txt

python train.py --epochs 50 --batch_size 16

## 🔧 Cài đặt

```
