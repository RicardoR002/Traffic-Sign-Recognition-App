# Traffic Sign Recognition App

A Streamlit-based web application that can classify traffic signs from images or camera input.

## Features

- Upload images for traffic sign classification
- Use camera input for real-time classification
- User-friendly interface
- Cloud deployment ready

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the App Locally

To run the app locally, use the following command:
```bash
streamlit run app.py
```

## Deployment to Streamlit Cloud

1. Create a GitHub repository and push your code
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Deploy the app

## Notes

- The current model is a placeholder. For production use, you should:
  - Train a model on the GTSRB (German Traffic Sign Recognition Benchmark) dataset
  - Save the trained model weights
  - Load the trained model instead of creating a new one
- The app supports 43 different traffic sign classes from the GTSRB dataset

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies 