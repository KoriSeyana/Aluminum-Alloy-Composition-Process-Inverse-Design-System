import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import shap
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


start = time.perf_counter()
plotconfig = False
filename = 'data.xlsx'  # Modify  dataset name
sheet = 'Sheet6'

# ==========================================
# Hyperparameter setting
# ==========================================
PARAMS = {
    # 1. Network architecture parameters
    'input_dim': 12,  # Number of input features
    'output_dim': 3,  # Number of output performance indicators

    'hidden_layers': [32, 64, 32],

    # 2. Training parameters
    'learning_rate': 0.001,
    'epochs': 2000,
    'batch_size': 32,
    'dropout_rate': 0.2,
    'weight_decay': 1e-4,
    'test_size': 0.2,
    #'seed': int(time.time())  #
    'seed': 20121221  # Fixed seed for reproducibility
}


torch.manual_seed(PARAMS['seed'])
np.random.seed(PARAMS['seed'])


# ==========================================
# data preparation
# ==========================================
def load_and_process_data():

    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' does not exist. Please ensure that the Excel file is in the same directory as the script.")
        exit()

    try:
        df = pd.read_excel(filename, sheet_name=sheet)
    except Exception as e:
        print(f"An error occurred while reading the Excel file.: {e}")
        exit()

    # -------------------------------------

    X = df.iloc[:, :PARAMS['input_dim']].values
    y = df.iloc[:, PARAMS['input_dim']:].values

    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_y = MinMaxScaler(feature_range=(-1, 1))

    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)


    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=PARAMS['test_size'], random_state=PARAMS['seed']
    )

    # Convert to PyTorch tensor
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_x, scaler_y


# ==========================================
# Build a neural network model
# ==========================================
class ForwardDNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout_rate):
        super(ForwardDNN, self).__init__()


        self.layers = nn.ModuleList()


        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Dropout(p=dropout_rate))  # Dropout


        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(p=dropout_rate))


        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ==========================================
# main training loop
# ==========================================
if __name__ == "__main__":

    X_train, y_train, X_test, y_test, scaler_x, scaler_y = load_and_process_data()


    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True)


    model = ForwardDNN(
        PARAMS['input_dim'],
        PARAMS['hidden_layers'],
        PARAMS['output_dim'],
        PARAMS['dropout_rate']
    )

    #  Define the loss function and the optimizer
    criterion = nn.MSELoss()  # MSE Loss
    optimizer = optim.Adam(model.parameters(), lr=PARAMS['learning_rate'], weight_decay=PARAMS['weight_decay'])

    #  training loop
    train_losses = []
    test_losses = []

    print("start training...")
    for epoch in range(PARAMS['epochs']):
        model.train()
        epoch_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)


        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = criterion(test_pred, y_test).item()
            test_losses.append(test_loss)

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/{PARAMS['epochs']}], Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}")

    # ==========================================
    # Result visualization and evaluation
    # ==========================================
    if plotconfig := True:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, 'b-o', markersize=2, linewidth=2, label='Train Loss')
        plt.plot(test_losses, 'r--s', markersize=2, linewidth=2, label='Test Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('MSE Loss')
        plt.legend(loc='upper left', fontsize=11)
        plt.grid(True, linestyle='--', color=[0.8, 0.8, 0.8])
        plt.savefig('Training and Validation Loss.png', dpi=300)
        plt.show()


    from sklearn.metrics import r2_score


    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy()

    y_test_real = scaler_y.inverse_transform(y_test.numpy())
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)



target_names = ['YS(MPa)', 'TS(MPa)', 'EL(%)']

print("\n--- Item-based indicator assessment ---")
for i in range(3):
    y_t = y_test_real[:, i]
    y_p = y_pred_real[:, i]

    _mse = mean_squared_error(y_test_real, y_pred_real)
    _rmse = rmse = np.sqrt(_mse)
    _mape = mean_absolute_percentage_error(y_t, y_p) * 100
    _r2 = r2_score(y_t, y_p)

    print(f"【{target_names[i]}】:")
    print(f"   RMSE: {_rmse:.2f} |MSE: {_mse:.2f}| MAPE: {_mape:.2f}% | R2: {_r2:.4f}")

print("\n--- Item-based indicator assessment complete ---")

# ==========================================
# SHAP
# ==========================================

print("\n---  SHAP starting ---")


background_size = 100
if len(X_train) > background_size:

    indices = np.random.choice(len(X_train), background_size, replace=False)
    background = X_train[indices]
else:
    background = X_train


test_size_shap = 50
if len(X_test) > test_size_shap:
    test_indices = np.random.choice(len(X_test), test_size_shap, replace=False)
    X_test_shap = X_test[test_indices]
else:
    X_test_shap = X_test


feature_names = [
    'Zn', 'Mg', 'Cu', 'Zr', 'Ti', 'Fe',
    'Si', 'Casting', 'Homogenization', 'Extrusion', 'Solution Treatment', 'Aging'
]


try:
    model.eval()
    explainer = shap.DeepExplainer(model, background)


    raw_shap_values = explainer.shap_values(X_test_shap, check_additivity=False)


    processed_shap_values = []


    def to_numpy(d):
        return d.cpu().detach().numpy() if hasattr(d, 'cpu') else np.array(d)


    vals = to_numpy(raw_shap_values)

    if isinstance(raw_shap_values, list):

        processed_shap_values = [to_numpy(v) for v in raw_shap_values]
    elif vals.ndim == 3:

        num_outputs = vals.shape[2]
        processed_shap_values = [vals[:, :, i] for i in range(num_outputs)]
    else:

        processed_shap_values = [vals]


    target_labels = ['YS(MPa)', 'TS(MPa)', 'EL(%)']


    X_test_display = scaler_x.inverse_transform(X_test_shap.numpy())


    for i in range(min(len(processed_shap_values), len(target_labels))):
        label = target_labels[i]
        print(f"正在生成 {label} 的 SHAP 分析图...")

        plt.figure()
        plt.title(f"{label} - Feature Importance Analysis", fontsize=14)


        shap.summary_plot(
            processed_shap_values[i],
            features=X_test_display,
            feature_names=feature_names,
            show=False,
            plot_type="dot"
        )

        plt.tight_layout()
        plt.savefig(f'SHAP_Analysis_{label}.png', dpi=300, bbox_inches='tight')
        # plt.show()
        plt.close()

    print("SHAP Analysis chart has been saved。")

except Exception as e:
    import traceback

    traceback.print_exc()
    print(f"SHAP The analysis still has some problems.: {e}")

#----------------------------------------------------------
#------------------Start reverse engineering----------------------------
#----------------------------------------------------------

print("\n--- Start training the reverse design model ---")
class InverseDesigner:
    def __init__(self, forward_model, scaler_x, scaler_y, bounds):
        """

        """
        self.model = forward_model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.bounds = bounds

        # Freeze the parameters of the forward model
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def design(self, target_properties, num_candidates=5, steps=1000, lr=0.1, a=1.3, b=1.3, c=0.1, d=0.01, e=0.5):
        """
        target_properties:
        num_candidates:
        """
        # 1. 目标值标准化
        target_scaled = self.scaler_y.transform([target_properties])
        target_tensor = torch.FloatTensor(target_scaled).repeat(num_candidates, 1)

        # 2. Random initialization of input X

        init_x = torch.randn(num_candidates, 12, requires_grad=True)

        # 3. Define the optimizer，
        optimizer = optim.Adam([init_x], lr=lr)

        print(f"Start reverse engineering，target: {target_properties}，parallel search {num_candidates} options...")

        for i in range(steps):
            optimizer.zero_grad()

            # --- Forward Pass ---
            predictions = self.model(init_x)



            # A. Performance error Loss (MSE)
            loss_performance = nn.MSELoss()(predictions, target_tensor)

            # Method 1: Simple Mean Square Error
            current_pred_scaled = self.model(init_x)
            diff = current_pred_scaled - target_tensor

            # Method 2: Weighted Difference
            weights = torch.tensor([1.5, 1.0, 0.8])  # Example weights for YS, TS, EL
            weighted_diff = (diff ** 2) * weights
            loss_performance_forward = torch.mean(weighted_diff)

            # B.  (Soft Constraint)

            loss_out_of_bound = torch.mean(torch.relu(torch.abs(init_x) - 3.0))

            # C. diversity Loss  - （Batch Optimization）
            # Simultaneously perform Adam optimization. And add a "diversity repulsion term" to the Loss, forcing these 10 sets of solutions to move away from each other and not converge to the same location.
            # We hope that the differences between the num_candidates solutions are as large as possible.
            # Calculate the distance between each pair, and the smaller the distance, the larger the Loss.

            if num_candidates > 1:
                # Calculate Distance Matrix
                dist_matrix = torch.cdist(init_x, init_x, p=2)
                # Add the identity matrix to prevent division by zero (the distance from oneself to oneself)
                dist_matrix = dist_matrix + torch.eye(num_candidates)

                loss_diversity = torch.sum(1.0 / dist_matrix)
            else:
                loss_diversity = 0.0

            # D. Integer Constraint
            integer_indices =[7, 8, 9, 10, 11]
            scaler_min = torch.FloatTensor(self.scaler_x.min_)
            scaler_scale = torch.FloatTensor(self.scaler_x.scale_)

            x_physical_grad = (init_x - scaler_min) / scaler_scale
            target_params = x_physical_grad[:, integer_indices]
            # 4. Calculate the sine penalty: sin(pi * x)^2，
            loss_integer = torch.mean(torch.sin(torch.pi * target_params) ** 2)


            # --- total Loss ---

            total_loss = (a * loss_performance + b * loss_performance_forward +
                          c * loss_out_of_bound + d * loss_diversity + e * loss_integer)

            total_loss.backward()
            optimizer.step()

            # ---  Physical Boundary Projection (Hard Constraint) ---
            # After each update, X must be brought back within the physically reasonable range.
            # This requires first de-standardizing -> truncating -> and then standardizing again.
            with torch.no_grad():
                # 1. Return to physical space
                x_physical = self.scaler_x.inverse_transform(init_x.detach().numpy())

                # 2. Implement Clamping
                for dim in range(12):
                    min_val, max_val = self.bounds[dim]
                    x_physical[:, dim] = np.clip(x_physical[:, dim], min_val, max_val)

                # 3. Return to the standardized space and reassign the values
                x_scaled_back = self.scaler_x.transform(x_physical)
                init_x.data = torch.FloatTensor(x_scaled_back)

            if (i + 1) % 200 == 0:
                print(f"Step {i + 1}: Loss = {total_loss.item():.4f} (Perf: {loss_performance.item():.4f})")

        # 4. Output the final result
        final_x_physical = self.scaler_x.inverse_transform(init_x.detach().numpy())

        integer_indices = [7, 8, 9, 10, 11]
        final_x_physical[:, integer_indices] = np.round(final_x_physical[:, integer_indices])

        final_x_physical[:, integer_indices] = np.maximum(final_x_physical[:, integer_indices], 0)

        final_x_scaled = self.scaler_x.transform(final_x_physical)
        final_x_tensor = torch.FloatTensor(final_x_scaled)

        with torch.no_grad():
            final_pred_scaled = self.model(final_x_tensor)

        final_pred_physical = self.scaler_y.inverse_transform(final_pred_scaled.numpy())

        # ==========================================================

        return final_x_physical, final_pred_physical


# ==========================================
#
# ==========================================
if __name__ == "__main__":

    bounds = [(5.57, 8.7), (1.91, 2.8), (0.5, 2.5), (0, 0.11), (0, 0.15), (0, 0.17), (0, 0.041),
              (0, 1), (0, 4), (0, 1), (0, 12), (0, 13)]
    X_train, y_train, X_test, y_test, scaler_x, scaler_y = load_and_process_data()
    designer = InverseDesigner(model, scaler_x, scaler_y, bounds)

    # target：
    target = [700, 700, 12]
    target_arr = np.array(target)

    designed_recipes, predicted_properties = designer.design(target, num_candidates=5)
    errors = np.mean(np.abs((predicted_properties - target_arr) / target_arr), axis=1) * 100
    sorted_indices = np.argsort(errors)


    print("\n=== Reverse design results ===")
    for i in range(5):
        print(f"\nproject {i + 1}:")
        print(f"  Design components / process: {designed_recipes[i]}")
        print(f"  predicted performance : {predicted_properties[i]}")
        print(f"  Target performance: {target}")

    top_n = 3
    best_indices = sorted_indices[:top_n]

    print(f"\n" + "=" * 40)
    print(f"  Filtering result: The best {top_n} options")
    print(f"   (Sorting criteria: The average relative error of the three indicators)")
    print("=" * 40)
    for rank, idx in enumerate(best_indices):

        recipe = designed_recipes[idx]
        props = predicted_properties[idx]
        err = errors[idx]


        print(f"\n rank No.{rank + 1} (Average error: {err:.4f}%)")
        print(f"----------------------------------------")
        print(f"   [Ingredients/Process]:")

        print(f"   {np.array2string(recipe, precision=4, suppress_small=True)}")

        print(f"   [Ingredients vs target]:")
        print(
            f"     YS: {props[0]:.1f} MPa (target {target[0]}) -> bias {(props[0] - target[0]) / target[0] * 100:.2f}%")
        print(
            f"     TS: {props[1]:.1f} MPa (target {target[1]}) -> bias {(props[1] - target[1]) / target[1] * 100:.2f}%")
        print(
            f"     EL  : {props[2]:.2f} %   (target {target[2]}) -> bias {(props[2] - target[2]) / target[2] * 100:.2f}%")


    best_recipe = designed_recipes[best_indices[0]]