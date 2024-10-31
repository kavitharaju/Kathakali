import cv2

import kathakali_mudra_recognition


image_path = "sample.jpg"  

result = kathakali_mudra_recognition.mudra_recognize(image_path)

print(result)