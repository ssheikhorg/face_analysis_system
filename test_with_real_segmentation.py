# test_with_real_segmentation.py
import httpx
import base64

# Read your actual images
with open("images/original_image.png", "rb") as f:
    real_image_b64 = base64.b64encode(f.read()).decode("utf-8")

with open("images/segmentation_map.png", "rb") as f:
    real_segmentation_b64 = base64.b64encode(f.read()).decode("utf-8")

# Use landmarks from your file
landmarks = [
    {"x": 482.6342021226883, "y": 732.6152205467224},
    {"x": 515.7728451490402, "y": 661.8982672691345},
    {"x": 507.5305451452732, "y": 680.4435551166534},
    {"x": 530.0111111998558, "y": 583.1576526165009},
    {"x": 525.0079814493656, "y": 640.3182983398438},
    {"x": 537.5344596505165, "y": 611.397922039032},
    {"x": 568.4228312373161, "y": 540.363273024559},
    {"x": 435.51949658989906, "y": 473.9514231681824},
    {"x": 590.5627065896988, "y": 488.33899796009064},
    {"x": 602.3263959288597, "y": 459.9365532398224},
    {"x": 646.0475490689278, "y": 353.96133959293365},
    {"x": 479.38548347353935, "y": 740.0785624980927},
    {"x": 476.2514768242836, "y": 747.2536146640778},
    {"x": 474.32571959495544, "y": 751.920211315155},
    {"x": 474.44620656967163, "y": 751.968663930893},
    {"x": 470.855353474617, "y": 760.6266021728516},
]

payload = {
    "image": real_image_b64,
    "landmarks": landmarks,
    "segmentation_map": real_segmentation_b64,
}

print("Testing with REAL segmentation map...")
print(f"Image size: {len(real_image_b64)} chars")
print(f"Segmentation size: {len(real_segmentation_b64)} chars")
print(f"Landmarks: {len(landmarks)} points")

response = httpx.post("http://localhost:8000/api/v1/frontal/crop/submit", json=payload)

print(f"Status: {response.status_code}")
result = response.json()
print(f"Contours generated: {len(result.get('result', {}).get('mask_contours', {}))}")

# Decode and check the SVG
import base64

svg_result = base64.b64decode(result.get("result", {}).get("svg", "")).decode("utf-8")
print(f"SVG content preview: {svg_result[:200]}...")
