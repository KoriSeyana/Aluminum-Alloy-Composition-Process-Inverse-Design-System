import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import os
import pandas as pd

from model_utils import PARAMS, ForwardDNN, set_seed, load_and_process_data


class InverseDesigner:
    def __init__(self, forward_model, scaler_x, scaler_y, bounds, dataset_X, dataset_y):

        self.model = forward_model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.bounds = bounds


        self.dataset_X = dataset_X
        self.dataset_y = dataset_y

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def design(self, target_properties, num_candidates=5, steps=3000, lr=0.1, a=0, b=1.5, c=0.1, d=0.01, e=0.5):
        # 1. Standardize target values
        target_scaled_np = self.scaler_y.transform([target_properties])
        target_tensor = torch.FloatTensor(target_scaled_np).repeat(num_candidates, 1)  # (N, 3)
        target_single = torch.FloatTensor(target_scaled_np)  # (1, 3) 用于搜索


        print("正在从历史数据中搜索最接近目标的配方作为起点...")


        # 1. Compute Euclidean distances from all historical points to the target
        distances = torch.norm(self.dataset_y - target_single, dim=1)

        # 2. Find indices of the nearest num_candidates points
        if len(distances) < num_candidates:
            top_k_indices = torch.argsort(distances)[:len(distances)]
            top_k_indices = torch.cat([top_k_indices] * (num_candidates // len(distances) + 1))[:num_candidates]
        else:
            _, top_k_indices = torch.topk(distances, num_candidates, largest=False)

        # 3. Select corresponding X (composition/process)
        best_starts = self.dataset_X[top_k_indices]  # (num_candidates, 12)

        # 4. Add small random Gaussian noise
        noise = 0.05 * torch.randn_like(best_starts)
        init_x_data = best_starts + noise

        # 5. Wrap as trainable parameters
        init_x = init_x_data.clone().detach().requires_grad_(True)
        # ============================================================


        optimizer = optim.Adam([init_x], lr=lr)


        print(
            f"Initialization complete. Base performance deviation range of starting recipes: {distances[top_k_indices].min().item():.4f} - {distances[top_k_indices].max().item():.4f} (Scaled Space)")

        # Pre-fetch scaler parameters
        scaler_min = torch.FloatTensor(self.scaler_x.min_)
        scaler_scale = torch.FloatTensor(self.scaler_x.scale_)
        integer_indices = [7, 8, 9, 10, 11]


        history_log = []

        for i in range(steps):
            optimizer.zero_grad()

            predictions = self.model(init_x)


            with torch.no_grad():
                # Get current predicted physical values
                curr_pred_scaled = predictions.detach().cpu().numpy()
                curr_pred_physical = self.scaler_y.inverse_transform(curr_pred_scaled)
                # Flatten and store in list
                history_log.append(curr_pred_physical.flatten())

            # A. Simple MSE
            loss_performance = nn.MSELoss()(predictions, target_tensor)

            # B. Weighted MSE
            diff = self.model(init_x) - target_tensor
            weights = torch.tensor([10.0, 5.0, 3])
            weighted_diff = (diff ** 2) * weights
            loss_performance_forward = torch.mean(weighted_diff)

            # C. Out-of-bound penalty
            loss_out_of_bound = torch.mean(torch.relu(torch.abs(init_x) - 1.05))

            # D. Diversity
            if num_candidates > 1:
                dist_matrix = torch.cdist(init_x, init_x, p=2) + torch.eye(num_candidates)
                loss_diversity = torch.sum(1.0 / dist_matrix)
            else:
                loss_diversity = 0.0

            # E. Integer constraint
            x_physical_grad = (init_x - scaler_min) / scaler_scale
            target_params = x_physical_grad[:, integer_indices]
            loss_integer = torch.mean(torch.sin(torch.pi * target_params) ** 2)

            # Dynamic e
            if i < 500:
                current_e = 0.0
            elif i < 2000:
                current_e = e * (i - 500) / 300
            else:
                current_e = e

            # F. Pull to center
            loss_center = torch.mean(init_x ** 2)
            f = 0.01

            total_loss = (a * loss_performance + b * loss_performance_forward +
                          c * loss_out_of_bound + d * loss_diversity +
                          current_e * loss_integer +
                          f * loss_center)

            total_loss.backward()
            optimizer.step()

            # Physical bounds projection
            with torch.no_grad():
                x_physical = self.scaler_x.inverse_transform(init_x.detach().numpy())
                for dim in range(12):
                    min_val, max_val = self.bounds[dim]
                    x_physical[:, dim] = np.clip(x_physical[:, dim], min_val, max_val)
                x_scaled_back = self.scaler_x.transform(x_physical)
                init_x.data = torch.FloatTensor(x_scaled_back)

            if (i + 1) % 200 == 0:
                print(f"Step {i + 1}: Total Loss = {total_loss.item():.4f}")

        # Final processing
        final_x_physical = self.scaler_x.inverse_transform(init_x.detach().numpy())
        final_x_physical[:, integer_indices] = np.round(final_x_physical[:, integer_indices])
        final_x_physical[:, integer_indices] = np.maximum(final_x_physical[:, integer_indices], 0)

        final_x_scaled = self.scaler_x.transform(final_x_physical)
        with torch.no_grad():
            final_pred_scaled = self.model(torch.FloatTensor(final_x_scaled))
        final_pred_physical = self.scaler_y.inverse_transform(final_pred_scaled.numpy())


        return final_x_physical, final_pred_physical, np.array(history_log)


def run_inverse_design(target_values=None):
    print("=" * 30)
    print(" >>> Entering Module 3: Inverse Design (Data Warm-start) <<<")
    print("=" * 30)

    if not os.path.exists('model_weights.pth'):
        print("Error: model file not found. Please run '1. Train Model' first.")
        return

    set_seed()

    try:
        scaler_x = joblib.load('scaler_x.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        model = ForwardDNN(PARAMS['input_dim'], PARAMS['hidden_layers'], PARAMS['output_dim'], PARAMS['dropout_rate'])
        model.load_state_dict(torch.load('model_weights.pth'))
    except Exception as e:
        print(f"Failed to load model files: {e}")
        return

    print("Loading dataset to find best starting points...")
    X_train, y_train, X_test, y_test, _, _ = load_and_process_data()
    dataset_X = torch.cat([X_train, X_test], dim=0)
    dataset_y = torch.cat([y_train, y_test], dim=0)
    print(f"Loaded historical data {len(dataset_X)} entries.")

    bounds = [(7.46,7.47), (2.35,2.41), (1.95,2.05), (0,0.12), (0.02,0.03), (0.1,0.12), (0.02,0.03),(0,1),(0,4),(0,1),(0,12),(0,13)]
    # bounds = [(5.57, 8.7), (1.91, 2.8), (0.5, 2.5), (0, 0.11), (0, 0.15), (0, 0.17), (0, 0.041),
    #          (0, 1), (0, 4), (0, 1), (0, 12), (0, 13)]

    if target_values is None:
        print("No input provided, using default target: YS=700, TS=700, EL=12")
        target = [700, 700, 12]
    else:
        target = target_values
        print(f"Received user target: {target}")

    designer = InverseDesigner(model, scaler_x, scaler_y, bounds, dataset_X, dataset_y)


    n_candidates = 5

    designed_recipes, predicted_properties, history_data = designer.design(target, num_candidates=n_candidates)

    print("\n>>> Exporting optimization trajectory data...")
    column_names = []
    prop_names = ['YS', 'TS', 'EL']
    for i in range(n_candidates):
        for prop in prop_names:
            column_names.append(f'Cand{i + 1}_{prop}')

    df_history = pd.DataFrame(history_data, columns=column_names)
    df_history.insert(0, 'Step', range(1, len(df_history) + 1))
    excel_filename = 'design_optimization_log.xlsx'
    df_history.to_excel(excel_filename, index=False)
    print(f"Successfully saved optimization trajectory to: {excel_filename}")

    target_arr = np.array(target)
    with np.errstate(divide='ignore', invalid='ignore'):
        errors = np.mean(np.abs((predicted_properties - target_arr) / target_arr), axis=1) * 100
        errors = np.nan_to_num(errors, nan=100.0, posinf=100.0, neginf=100.0)

    sorted_indices = np.argsort(errors)
    top_n = 3
    print(f"\nSelection results: Top {top_n} solutions (Metric: MAPE)")
    print("=" * 40)

    for rank, idx in enumerate(sorted_indices[:top_n]):
        recipe = designed_recipes[idx]
        props = predicted_properties[idx]
        err = errors[idx]

        print(f"\n🏆 Rank No.{rank + 1} (Mean error: {err:.4f}%)")
        print(f"----------------------------------------")
        print(f"   [Recipe]: {np.array2string(recipe, precision=4, suppress_small=True)}")
        print(f"   [Prediction]: YS={props[0]:.1f} MPa, TS={props[1]:.1f} MPa, EL={props[2]:.2f} %")
        print(f"   [Target]: YS={target[0]:.1f} MPa, TS={target[1]:.1f} MPa, EL={target[2]:.2f} %")