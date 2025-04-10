# Streamlit_APP

A sleek, interactive web application built with Streamlit to classify images from a Fashion MNIST-like dataset using a neural network with hyperparameter tuning via Optuna. Upload your CSV, train the model, and visualize the results—all with a modern, dashy UI!

🚀 Features
CSV Upload: Upload a Fashion MNIST-like CSV file (1 label + 784 pixel columns).
Data Preview: View the first few rows and a 4x4 grid of the first 16 images.
Neural Network Training: Train a custom PyTorch neural network with batch normalization, dropout, and ReLU activations.
Hyperparameter Tuning: Optimize model parameters using Optuna with customizable trial counts.
Visualizations: Explore training results with Optuna plots (Optimization History, Parallel Coordinate, Slice, Contour, Parameter Importances).
Dashy UI: A modern interface with gradient backgrounds, card layouts, custom buttons, and a progress bar.
Device Support: Automatically uses CUDA if available, falls back to CPU otherwise.


.
📊 How It Works
Data Processing: Loads and normalizes the CSV data, splitting it into train/test sets.
Model: A PyTorch neural network with configurable hidden layers, neurons, and dropout.
Optimization: Optuna tunes hyperparameters like learning rate, batch size, and optimizer type.
Evaluation: Computes accuracy on the test set.
Visualization: Displays data previews and Optuna plots in a styled UI.

🧩 Dependencies
streamlit: Web app framework
pandas: Data handling
scikit-learn: Train-test split
torch: Neural network framework
optuna: Hyperparameter optimization
matplotlib: Image grid plotting
plotly & kaleido: Optuna visualizations

⚙️ Configuration
File Format: CSV with 785 columns (1 label + 784 pixels).
Hyperparameters: Tuned via Optuna (e.g., layers, neurons, epochs, learning rate).
Device: Auto-detects CUDA or defaults to CPU.
