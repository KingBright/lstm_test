import joblib
import config # Ensure this imports the correct config version
import numpy as np
import os
# *** ADD THIS IMPORT LINE ***
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# --- Ensure path matches the scaler from the last run ---
TARGET_SCALER_PATH = config.TARGET_SCALER_PATH
# ---

print(f"正在加载 Target Scaler: {TARGET_SCALER_PATH}")

if os.path.exists(TARGET_SCALER_PATH):
    try:
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        print("Scaler 加载成功!")
        print(f"Scaler 类型: {type(target_scaler)}")

        # Now isinstance checks will work
        if isinstance(target_scaler, MinMaxScaler):
            print(f"  Feature Range: {target_scaler.feature_range}")
            print(f"  Data Min (theta, theta_dot): {target_scaler.data_min_}")
            print(f"  Data Max (theta, theta_dot): {target_scaler.data_max_}")
            print(f"  Scale (theta, theta_dot): {target_scaler.scale_}")
            print(f"  Min Offset (theta, theta_dot): {target_scaler.min_}")
            scaled_zero = target_scaler.transform(np.array([[0.0, 0.0]]))
            print(f"  Scaled value for [0.0, 0.0]: {scaled_zero}")
            unscaled_zero = target_scaler.inverse_transform(scaled_zero)
            print(f"  Inverse transformed scaled zero: {unscaled_zero}")
        elif isinstance(target_scaler, StandardScaler):
             print(f"  Mean (theta, theta_dot): {target_scaler.mean_}")
             print(f"  Scale (Std Dev) (theta, theta_dot): {target_scaler.scale_}")
             scaled_zero = target_scaler.transform(np.array([[0.0, 0.0]]))
             print(f"  Scaled value for [0.0, 0.0]: {scaled_zero}")
             unscaled_zero = target_scaler.inverse_transform(scaled_zero)
             print(f"  Inverse transformed scaled zero: {unscaled_zero}")
        else:
            print("未知的 Scaler 类型，无法打印详细信息。")

    except Exception as e:
        print(f"加载或检查 Scaler 时出错: {e}")
else:
    print(f"错误: Scaler 文件未找到: {TARGET_SCALER_PATH}")