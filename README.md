# AI-Food-Recognition-and-Nutrition-Analyzer
Recognition of food and tell about its nutrition, fibre.
# 🍽️ Fruit & Vegetable Detection with Nutritional Insights

A computer vision project using YOLOv5 to detect multiple fruits and vegetables in a single image and return their respective nutritional information. Ideal for dietary analysis and educational demonstrations.

## 📸 Demo

<p align="center">
  <img src="demo_image.jpg" alt="Demo Image with Detections" width="400"/>
</p>

## 🔍 Features

- ✅ Detects multiple items on a plate using a custom-trained YOLOv5 model
- ✅ Annotates images with bounding boxes and confidence scores
- ✅ Displays nutritional values (Calories, Fiber, Vitamin C) for each detected item
- ✅ Includes demo fallback for showcasing even with low training epochs

## 🧠 Model Details

- Framework: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- Training Epochs: 39 (to be extended)
- Format: `best.pt` checkpoint trained on a custom dataset of fruits and vegetables
- Confidence threshold: dynamically adjustable (default: 0.2 for final prediction)

## 📁 Project Structure


## 🧪 How It Works

1. Load an image using PIL.
2. Run detection using YOLOv5.
3. For each object with confidence ≥ 0.2:
   - Draw bounding box
   - Display label and confidence
   - Fetch nutrition information from dictionary
4. Optionally use demo-only hardcoded boxes for presentation.

## 🥗 Nutrition Dataset

Stored internally as a Python dictionary with the following nutrients:

- Calories
- Fiber (g)
- Vitamin C (mg)

> Example:
```python
'nutrition_info = {
    'banana': {'Calories': 89, 'Fiber': 2.6, 'Vitamin C': 8.7},
    'potato': {'Calories': 77, 'Fiber': 2.2, 'Vitamin C': 19.7},
    ...
}
# Clone repository
git clone https://github.com/yourusername/fruit-veg-detector.git
cd fruit-veg-detector

# Install dependencies
pip install -r requirements.txt

# Optional: Install YOLOv5 dependencies
pip install -U torch torchvision
