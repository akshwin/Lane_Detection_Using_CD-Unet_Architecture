
# 🚦 Cascaded U-Net Lane Detection Web App

A real-time web application to detect road lanes from images using a cascaded U-Net architecture. Built with Flask, OpenCV, and deployed on Render.

🌐 **Live Demo:**  
[![Live Demo](https://img.shields.io/badge/Visit%20App-Click%20Here-blue?style=for-the-badge)](https://lane-detection-cascaded-unet.onrender.com/)

---

## 🚀 Features

- Upload road images (JPEG/PNG).
- Predict lane segmentation using a U-Net-based cascaded deep learning model.
- View original and predicted images side by side.
- Clean, responsive UI with Bootstrap.
- Deployed on Render using Flask + Gunicorn.

---

## 🛠️ Tech Stack

- Python 3.x
- Flask + Gunicorn
- HTML, CSS, Bootstrap
- OpenCV, NumPy, Pillow
- Render (Cloud Deployment)

---

## 📁 Project Structure

```
lane-detection-app/
├── app.py
├── templates/
│   └── index.html
├── uploads/
│   └── (uploaded images)
├── predictions/
│   └── (output lane mask images)
├── requirements.txt
├── README.md
```

---

## ⚙️ Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/lane-detection-app.git
cd lane-detection-app
```

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Locally

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

---

## 🖼️ Usage

1. Click **Choose Image** to upload a road scene image.
2. Click **Predict Lane** to generate the lane mask.
3. View the input and output images side-by-side.

---

## 🌍 Deployment on Render

The app is deployed at:  
🔗 **[https://lane-detection-cascaded-unet.onrender.com](https://lane-detection-cascaded-unet.onrender.com)**

### Render Configuration Tips

- **Start Command:**
  ```bash
  gunicorn app:app
  ```
- **Build Command:**
  ```bash
  pip install -r requirements.txt
  ```
- **Python Version:**
  Use `runtime.txt` or set in `render.yaml`.

---

## 📦 Requirements

`requirements.txt` should contain:

```
Flask
gunicorn
numpy
opencv-python
Pillow
```

Include any model-specific libraries as needed.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Author

Made with 💡 by **Akshwin T**  
🔗 [LinkedIn](https://www.linkedin.com/in/akshwin/) | [GitHub](https://github.com/akshwin)