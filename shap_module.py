import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import traceback

from model_utils import PARAMS, ForwardDNN, load_and_process_data, set_seed


def run_shap():
    print("=" * 30)
    print(" >>>  SHAP   <<<")
    print("=" * 30)

    if not os.path.exists('model_weights.pth'):
        print("Error: Model file not found!")
        return

    set_seed()


    try:
        scaler_x = joblib.load('scaler_x.pkl')
        print("Successfully loaded scaler_x.pkl")
    except:
        print("Warning: Unable to load scaler_x.pkl. Please try to obtain it from load_and_process_data....")

        X_train, y_train, X_test, y_test, scaler_x_func, _ = load_and_process_data()
        scaler_x = scaler_x_func


    X_train, y_train, X_test, y_test, _, _ = load_and_process_data()


    model = ForwardDNN(PARAMS['input_dim'], PARAMS['hidden_layers'], PARAMS['output_dim'], PARAMS['dropout_rate'])
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()


    background_size = 200
    if len(X_train) > background_size:
        indices = np.random.choice(len(X_train), background_size, replace=False)
        background = X_train[indices]
    else:
        background = X_train


    test_size_shap = 200
    if len(X_test) > test_size_shap:
        test_indices = np.random.choice(len(X_test), test_size_shap, replace=False)
        X_test_shap = X_test[test_indices]
    else:
        X_test_shap = X_test

    feature_names = [
        'Zn', 'Mg', 'Cu', 'Zr', 'Ti', 'Fe',
        'Si', 'Casting', 'Homogenization', 'Extrusion', 'Solution Treatment', 'Aging'
    ]
    target_labels = ['YS(MPa)', 'TS(MPa)', 'EL(%)']


    try:
        device = torch.device("cpu")
        model.to(device)
        background = background.to(device)
        X_test_shap = X_test_shap.to(device)

        print("Calculating SHAP values...")
        explainer = shap.DeepExplainer(model, background)
        raw_shap_values = explainer.shap_values(X_test_shap, check_additivity=False)


        processed_shap_values = []

        if isinstance(raw_shap_values, list):
            for v in raw_shap_values:
                if hasattr(v, 'detach'):
                    arr = v.detach().cpu().numpy()
                else:
                    arr = np.array(v)
                processed_shap_values.append(arr)
        else:
            if hasattr(raw_shap_values, 'detach'):
                vals = raw_shap_values.detach().cpu().numpy()
            else:
                vals = np.array(raw_shap_values)

            if vals.ndim == 3:
                processed_shap_values = [vals[:, :, i] for i in range(vals.shape[-1])]
            else:
                processed_shap_values = [vals]


        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 12


        X_display_arr = X_test_shap.cpu().numpy()


        X_test_display = scaler_x.inverse_transform(X_display_arr)

        print(f"Shape of the feature matrix: {X_test_display.shape}")

        loop_count = min(len(processed_shap_values), len(target_labels))

        for i in range(loop_count):
            label = target_labels[i]
            shap_val = processed_shap_values[i]

            shap_val = np.squeeze(shap_val)

            if shap_val.ndim == 3:
                if shap_val.shape[-1] == 3:
                    shap_val = shap_val[:, :, i]
                else:
                    shap_val = shap_val[:, :, 0]

            if shap_val.shape != X_test_display.shape:
                print(f" Skipping {label}: The shape of SHAP values {shap_val.shape} does not match the shape of the feature {X_test_display.shape}.")
                continue

            print(f"Generating the SHAP plot for {label}...")

            plt.figure()
            plt.title(f"{label} - Feature Importance")

            shap.summary_plot(
                shap_val,
                features=X_test_display,
                feature_names=feature_names,
                show=False,
                plot_type="dot"
            )

            save_name = f'SHAP_Analysis_{label}.png'
            plt.tight_layout()
            plt.savefig(save_name, dpi=600, bbox_inches='tight')
            plt.close()
            print(f" Chart has been saved:{save_name}")

        print("The SHAP analysis has been completed.")

    except Exception as e:
        print("\n Still an error occurred. Please check the data.")
        traceback.print_exc()