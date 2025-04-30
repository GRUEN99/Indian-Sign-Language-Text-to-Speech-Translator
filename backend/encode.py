import base64

with open(r"E:\project\backend\test_img.jpg", "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode('utf-8')
    print(encoded)
