```python
def hsv_processing(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv_img_height = hsv_img.shape[0]
    hsv_img_width = hsv_img.shape[1]

    # HSV range H,[0,360); S,[0,1); V,[0,1)
    # opencv HSV range h,[0,180); s,[0,255); v,[0,255)

    h_quantized = np.zeros((hsv_img_height, hsv_img_width), dtype=np.uint8)
    s_quantized = np.zeros((hsv_img_height, hsv_img_width), dtype=np.uint8)
    v_quantized = np.zeros((hsv_img_height, hsv_img_width), dtype=np.uint8)

    h = hsv_img[:, :, 0]
    s = hsv_img[:, :, 1]
    v = hsv_img[:, :, 2]

    h = 2 * h
    h_quantized[(h > 315) | (h <= 200)] = 0
    h_quantized[(h > 20) & (h <= 40)] = 1
    h_quantized[(h > 40) & (h <= 75)] = 2
    h_quantized[(h > 75) & (h <= 155)] = 3
    h_quantized[(h > 155) & (h <= 190)] = 4
    h_quantized[(h > 190) & (h <= 270)] = 5
    h_quantized[(h > 270) & (h <= 295)] = 6
    h_quantized[(h > 295) & (h <= 315)] = 7

    # 255*0.2 =51; 255*0.7=178
    s_quantized[(s <= 51)] = 0
    s_quantized[(s > 51) & (s <= 178)] = 1
    s_quantized[(s > 178)] = 2

    v_quantized[(v <= 51)] = 0
    v_quantized[(v > 51) & (v <= 178)] = 1
    v_quantized[(v > 178)] = 2

    final_score = 9 * h_quantized + 3 * s_quantized + v_quantized
    hist = cv2.calcHist([final_score], [0], None, [72], [0, 71]) / (hsv_img_height * hsv_img_width)
    # this hist is normalized
    return hist
```

