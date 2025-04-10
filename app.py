import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_slice, plot_contour, plot_param_importances
import io
import base64
import time


# Streamlit page config
st.set_page_config(page_title="Fashion MNIST Classifier", layout="wide", initial_sidebar_state="expanded")
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.write(f"Using device: {device}")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 2.5em;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subtitle {
        font-size: 1.5em;
        color: #34495e;
        margin-top: 1em;
    }
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stFileUploader {
        background-color: #ecf0f1;
        border-radius: 10px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    n_trials = st.slider("Number of Optuna Trials", 1, 20, 5, help="More trials = better tuning, but slower")
    st.info(f"Device: {torch.cuda.is_available() and 'CUDA' or 'CPU'}")

# Main content
st.markdown('<div class="title">👗 Image Dataset Analysis </div>', unsafe_allow_html=True)
st.markdown("Upload Your CSV file to train a neural network and visualize the results!", unsafe_allow_html=True)

# File uploader in a card
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📂 Upload CSV", type=['csv'], help="CSV with 'label' and 784 pixel columns")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        # Read and process the file
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File '{uploaded_file.name}' uploaded!")

        # Validate data
        if df.shape[1] != 785:
            st.error("❌ Invalid CSV format. Expected 785 columns (1 label + 784 pixels).")
        else:
            # Data preview and images in columns
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown('<div class="card"><h3>Data Preview</h3>', unsafe_allow_html=True)
                st.dataframe(df.head(), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="card"><h3>First 16 Images</h3>', unsafe_allow_html=True)
                fig, axes = plt.subplots(4, 4, figsize=(10, 10))
                for i, ax in enumerate(axes.flat):
                    img = df.iloc[i, 1:].values.reshape(28, 28)
                    ax.imshow(img, cmap='gray')
                    ax.axis('off')
                    ax.set_title(f"L: {df.iloc[i, 0]}", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            # Train-test split
            X = df.iloc[:, 1:].values / 255.0
            y = df.iloc[:, 0].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Custom Dataset
            class CustomDataset(Dataset):
                def __init__(self, features, labels):
                    self.features = torch.tensor(features, dtype=torch.float32)
                    self.labels = torch.tensor(labels, dtype=torch.long)
                def __len__(self): return len(self.features)
                def __getitem__(self, index): return self.features[index], self.labels[index]

            train_dataset = CustomDataset(X_train, y_train)
            test_dataset = CustomDataset(X_test, y_test)

            # Neural Network Model
            class MyNN(nn.Module):
                def __init__(self, input_dim, output_dim, num_hidden_layers, neurons_per_layer, dropout_rate):
                    super().__init__()
                    layers = []
                    for i in range(num_hidden_layers):
                        layers.append(nn.Linear(input_dim, neurons_per_layer))
                        layers.append(nn.BatchNorm1d(neurons_per_layer))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout_rate))
                        input_dim = neurons_per_layer
                    layers.append(nn.Linear(neurons_per_layer, output_dim))
                    self.model = nn.Sequential(*layers)
                def forward(self, x): return self.model(x)

            # Optuna Objective Function
            def objective(trial):
                num_hidden_layers = trial.suggest_int("num_hidden_layers", 1, 5)
                neurons_per_layer = trial.suggest_int("neurons_per_layer", 8, 128, step=8)
                epochs = trial.suggest_int("epochs", 10, 50, step=10)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
                dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
                batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
                optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'SGD', 'RMSprop'])
                weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

                model = MyNN(784, 10, num_hidden_layers, neurons_per_layer, dropout_rate).to(device)
                criterion = nn.CrossEntropyLoss()
                if optimizer_name == 'Adam':
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                elif optimizer_name == 'SGD':
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                else:
                    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

                model.train()
                for epoch in range(epochs):
                    for batch_features, batch_labels in train_loader:
                        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                        outputs = model(batch_features)
                        loss = criterion(outputs, batch_labels)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                model.eval()
                total, correct = 0, 0
                with torch.no_grad():
                    for batch_features, batch_labels in test_loader:
                        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                        outputs = model(batch_features)
                        _, predicted = torch.max(outputs, 1)
                        total += batch_labels.shape[0]
                        correct += (predicted == batch_labels).sum().item()
                return correct / total

            # Visualizations
            st.markdown('<div class="subtitle">Optuna Visualizations</div>', unsafe_allow_html=True)
            viz_options = ["Optimization History", "Parallel Coordinate", "Slice Plot", "Contour Plot", "Param Importances"]
            selected_viz = st.multiselect("Select Visualizations", viz_options, default=["Optimization History", "Param Importances"])
            # Training button
            if st.button("🚀 Start Training"):
                st.markdown('<div class="subtitle">Training Results</div>', unsafe_allow_html=True)
                with st.spinner("Optimizing hyperparameters..."):
                    progress_bar = st.progress(0)
                    study = optuna.create_study(direction='maximize')
                    #study.optimize(objective, n_trials=n_trials)
                    for i in range(n_trials):
                        study.optimize(objective, n_trials=1)
                        progress_bar.progress((i + 1) / n_trials)
                        time.sleep(0.1)  # Simulate processing time
                    st.success("Training completed!")
                
                # Display results in a card
                with st.container():
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write("**Best Parameters:**", study.best_params)
                    st.write("**Best Accuracy:**", f"{study.best_value:.4f}")
                    st.markdown('</div>', unsafe_allow_html=True)

                def save_plot_to_image(plot_func, filename):
                    fig = plot_func(study)
                    fig.write_image(filename)
                    with open(filename, "rb") as f:
                        return base64.b64encode(f.read()).decode()

                for viz in selected_viz:
                    with st.container():
                        st.markdown(f'<div class="card"><h3>{viz}</h3>', unsafe_allow_html=True)
                        if viz == "Optimization History":
                            img = save_plot_to_image(plot_optimization_history, "opt_history.png")
                        elif viz == "Parallel Coordinate":
                            img = save_plot_to_image(plot_parallel_coordinate, "parallel_coord.png")
                        elif viz == "Slice Plot":
                            img = save_plot_to_image(plot_slice, "slice.png")
                        elif viz == "Contour Plot":
                            img = save_plot_to_image(plot_contour, "contour.png")
                        elif viz == "Param Importances":
                            img = save_plot_to_image(plot_param_importances, "param_importance.png")
                        st.image(f"data:image/png;base64,{img}", use_column_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                torch.cuda.empty_cache()

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.write("Please ensure your CSV matches the Fashion MNIST format (785 columns).")
else:
    st.info("⏳ Waiting for file upload...")

# Footer
st.markdown('<hr><p style="text-align: center; color: #7f8c8d;">Built with ❤️ using Streamlit</p>', unsafe_allow_html=True)
