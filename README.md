# 🎨 SAM Image Segmentation Studio

A modern, AI-powered image segmentation application built with **Segment Anything Model (SAM)** and a beautiful **Gradio UI**.

![SAM Studio](https://img.shields.io/badge/AI-Powered-blue) ![Python](https://img.shields.io/badge/Python-3.8+-green) ![SAM](https://img.shields.io/badge/SAM-Model-orange)

## ✨ Features

- **🎯 Interactive Segmentation**: Point-and-click or box selection for precise object segmentation
- **🤖 AI Object Recognition**: Automatic object classification using MobileNet/ResNet
- **🎨 Modern UI**: Beautiful, responsive interface with animations and gradients
- **📥 Download Options**: Export masks and segmented objects in multiple formats
- **⚡ Real-time Preview**: See your selections before segmentation
- **🔄 GPU Acceleration**: CUDA support for faster processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- SAM model checkpoint file (`sam_vit_b_01ec64.pth`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/linga1010/SamPhoto.git
   cd SamPhoto
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download SAM model**
   - Download `sam_vit_b_01ec64.pth` from [SAM GitHub](https://github.com/facebookresearch/segment-anything)
   - Place it in the project root directory

4. **Run the application**
   ```bash
   python app.py
   ```

## 🎮 How to Use

1. **Upload Image**: Click on the upload area and select your image
2. **Choose Method**: Select "Point" or "Box" segmentation mode
3. **Make Selection**:
   - **Point Mode**: Click once on the object you want to segment
   - **Box Mode**: Click two corners to define a bounding box
4. **Preview**: Use "Preview Selection" to see your selection markers
5. **Segment**: Click "🚀 Segment Object" to perform AI segmentation
6. **Download**: Use the download section to save masks and objects

## 🛠️ Technical Stack

- **SAM (Segment Anything)**: Meta's foundation model for image segmentation
- **CLIP**: OpenAI's model for text-image understanding
- **Gradio**: Modern web UI framework
- **PyTorch**: Deep learning framework
- **OpenCV**: Computer vision operations
- **PIL/Pillow**: Image processing

## 📁 Project Structure

```
SamPhoto/
├── app.py              # Main application with UI
├── main.py             # Template file
├── requirements.txt    # Python dependencies
├── .gitignore         # Git ignore rules
├── README.md          # This file
└── sam_vit_b_01ec64.pth  # SAM model (download separately)
```

## ⚙️ Configuration

Key settings in `app.py`:

```python
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"  # SAM model path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRELOAD_MODELS_AT_STARTUP = True  # Faster startup
USE_SMALL_CLASSIFIER = True  # MobileNet vs ResNet
```

## 🎨 UI Features

- **Gradient Backgrounds**: Beautiful color transitions
- **Card-based Layout**: Clean, organized sections
- **Hover Animations**: Interactive feedback
- **Responsive Design**: Works on different screen sizes
- **Preview Images**: See results before downloading

## 📥 Download Formats

- **Mask**: Binary mask (white object on black background)
- **Object on Black**: Segmented object on black background
- **Cropped Transparent**: Object with transparent background

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [Gradio](https://gradio.app/) for the amazing UI framework

## 📞 Support

If you have any questions or issues, please open an issue on GitHub.

---

⭐ **Star this repository if you found it helpful!**