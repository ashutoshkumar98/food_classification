# 🍛 Indian Food Image Classifier

A deep learning project using **MobileNetV2** to classify Indian dishes from images. Trained on a rich dataset of 75+ Indian foods including *biryani*, *gulab jamun*, *dal makhani*, and more. Perfect for food recognition applications, recipe apps, and smart kitchen assistants.

---

## 📂 Dataset

- **Source**: [Kaggle - Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset)
- **Classes**: 75+
- **Structure**:
  ```
  dataset/
    biryani/
      img1.jpg
      img2.jpg
    rasgulla/
      img1.jpg
      ...
  ```

---

## 🛠️ Tech Stack

- **Language**: Python 3.7+
- **Frameworks & Libraries**:
  - TensorFlow 2.x
  - Keras
  - NumPy
  - Pillow (PIL)

---

## 🖥️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/ashutoshkumar98/food_classification.git
cd indian-food-classifier
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy pillow
```

### 3. Download Dataset

Download and extract the dataset from [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset) into a folder named `dataset` in the root directory.

---

## 🚀 Train the Model

Train a MobileNetV2 model using the images:

```bash
python train_model.py
```

- Saves model as `indian_food_model.h5`
- Uses `ImageDataGenerator` for augmentation
- 80/20 split for training and validation

---

## 🔍 Predict a Food Image

1. Place your test image in the root folder (e.g., `test4.jpg`)
2. Run:

```bash
python predict.py
```

- Requires:
  - `indian_food_model.h5`
  - `dataset/` for class label mapping

---

## 🧠 Model Architecture

- **Base**: MobileNetV2 (frozen)
- **Top Layers**:
  - `GlobalAveragePooling2D`
  - `Dense (128, ReLU)`
  - `Dropout (0.3)`
  - `Dense (Softmax)` – number of units = number of classes

---

## 📜 Supported Foods

Includes 75+ Indian food categories such as:

```
biryani, rasgulla, dal_makhani, aloo_gobi, naan, butter_chicken, jalebi, modak, phirni, paneer_butter_masala
```

See `List of Indian Foods.txt` for full list.

---

## 📱 Device Compatibility

Works on:

- Windows
- macOS
- Linux (Ubuntu)
- Raspberry Pi (via TensorFlow Lite – not included here)

---

## 💡 Future Enhancements

- TensorFlow Lite conversion
- Web UI using Streamlit or Flask
- Mobile app integration (React Native / Flutter)

---

## 📬 Contact

For questions or contributions, feel free to reach out at:

📧 **ashutoshkumar98998@gmail.com**

---

## 🤝 Acknowledgements

- Dataset by [Sourav Banerjee](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset)
- Model powered by [TensorFlow](https://www.tensorflow.org/) and [Keras](https://keras.io/)
