import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import time


from model_utils import PARAMS, ForwardDNN, load_and_process_data, set_seed


def run_training():
    print("=" * 30)
    print(" >>> Model Training <<<")
    print("=" * 30)

    set_seed()


    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False


    X_train, y_train, X_test, y_test, scaler_x, scaler_y = load_and_process_data()


    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True)


    model = ForwardDNN(PARAMS['input_dim'], PARAMS['hidden_layers'], PARAMS['output_dim'], PARAMS['dropout_rate'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=PARAMS['learning_rate'], weight_decay=PARAMS['weight_decay'])


    train_losses = []
    test_losses = []
    print("start training...")
    start_time = time.perf_counter()

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

    print(f"Training completed. Time taken: {time.perf_counter() - start_time:.2f}s")


    torch.save(model.state_dict(), 'model_weights.pth')
    joblib.dump(scaler_x, 'scaler_x.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("The model weights and the Scaler have been saved locally.")


    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('Training_Loss.png', dpi=300)
    print("The loss curve has been saved.")


    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test).numpy()

    y_test_real = scaler_y.inverse_transform(y_test.numpy())
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled)
    target_names = ['YS(MPa)', 'TS(MPa)', 'EL(%)']

    print("\n--- Final evaluation indicators ---")
    for i in range(3):
        y_t = y_test_real[:, i]
        y_p = y_pred_real[:, i]
        _rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        _rmse_item = np.sqrt(mean_squared_error(y_t, y_p))
        _mape = mean_absolute_percentage_error(y_t, y_p) * 100
        _r2 = r2_score(y_t, y_p)
        print(f"【{target_names[i]}】 RMSE: {_rmse_item:.2f} | MAPE: {_mape:.2f}% | R2: {_r2:.4f}")