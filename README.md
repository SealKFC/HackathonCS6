Below is a more concise README file for your "Get Swole" project:

---

# Get Swole

Get Swole is a fullstack web application that boosts gym performance by analyzing your form in real-time. It leverages MediaPipe for pose landmark detection and a custom model to verify full exercise repetitions.

## Features

- **Real-Time Pose Estimation:** Uses [MediaPipe](https://github.com/google/mediapipe) for extracting pose landmarks.
- **Full Rep Detection:** Combines angle calculations with a separately trained model.
- **Integrated Fullstack:** Seamlessly connects the frontend, backend, and ML components.

## Installation

### Prerequisites

- **Node.js** (v14+)
- **Python** (v3.8+)

### MediaPipe Setup

Follow the official [MediaPipe documentation](https://google.github.io/mediapipe/) for installation instructions.

### Backend

```bash
git clone https://github.com/yourusername/getswole.git
cd getswole/backend
npm install
cp .env.example .env
npm start
```

### Frontend

```bash
cd ../frontend
npm install
npm start
```

### Python Model

```bash
cd ../models
pip install -r requirements.txt
# Download the pretrained model (if available)
wget https://example.com/path-to-your-model
python run_model.py
```

## Usage

Open [http://localhost:3000](http://localhost:3000), allow camera access, and start your workout session while receiving real-time feedback.

## Contributing

Fork the repository, create a new branch for your changes, and submit a pull request.

## License

This project is licensed under the MIT License.

---
