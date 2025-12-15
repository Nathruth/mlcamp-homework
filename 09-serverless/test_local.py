# test_local.py
from lambda_handler import predict

img_url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

prob = predict(img_url)
print("Predicted probability:", prob)
