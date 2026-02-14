import numpy as np

def generate_true_motion(dt=0.1, T=60):
    time = np.arange(0, T, dt)
    
    x_true = []
    y_true = []
    theta_true = []
    omega_true = []
    velocity_true = []
    
    x = 0
    y = 0
    theta = 0
    velocity = 2
    
    for t in time:
        if 20 < t < 40:
            omega = 0.1
        else:
            omega = 0
        
        theta = theta + omega * dt
        x = x + velocity * np.cos(theta) * dt
        y = y + velocity * np.sin(theta) * dt
        
        x_true.append(x)
        y_true.append(y)
        theta_true.append(theta)
        omega_true.append(omega)
        velocity_true.append(velocity)
    
    return (
        time,
        np.array(x_true),
        np.array(y_true),
        np.array(theta_true),
        np.array(omega_true),
        np.array(velocity_true)
    )


def generate_gps(x_true, y_true, time, noise_std=1.5, dropout=True):
    gps_x = x_true + np.random.normal(0, noise_std, len(x_true))
    gps_y = y_true + np.random.normal(0, noise_std, len(y_true))
    
    if dropout:
        for i, t in enumerate(time):
            if 20 < t < 40:
                gps_x[i] = np.nan
                gps_y[i] = np.nan
    
    return gps_x, gps_y


def generate_imu(omega_true, time, noise_std=0.01):
    bias_drift = np.linspace(0, 0.02, len(time))
    imu_omega = omega_true + bias_drift + np.random.normal(0, noise_std, len(time))
    return imu_omega


def generate_wheel_velocity(velocity_true, noise_std=0.2):
    wheel_velocity = velocity_true + np.random.normal(0, noise_std, len(velocity_true))
    return wheel_velocity


def dead_reckoning(time, imu_omega, wheel_velocity, dt=0.1):
    
    x_est = []
    y_est = []
    theta_est = []
    
    x = 0
    y = 0
    theta = 0
    
    for i in range(len(time)):
        
        # Update heading using IMU
        theta = theta + imu_omega[i] * dt
        
        # Update position using wheel velocity
        x = x + wheel_velocity[i] * np.cos(theta) * dt
        y = y + wheel_velocity[i] * np.sin(theta) * dt
        
        x_est.append(x)
        y_est.append(y)
        theta_est.append(theta)
    
    return np.array(x_est), np.array(y_est), np.array(theta_est)


def ekf_fusion(time, imu_omega, wheel_velocity, gps_x, gps_y, dt=0.1):
    
    n = len(time)
    
    # State: [x, y, theta]
    x_est = np.zeros((n, 3))
    
    # Initial covariance
    P = np.eye(3) * 1.0
    
    # Process noise
    Q = np.diag([0.1, 0.1, 0.01])
    
    # Measurement noise (GPS)
    R = np.diag([2.0, 2.0])
    
    for i in range(1, n):
        
        # ===== Prediction Step =====
        theta = x_est[i-1, 2]
        
        theta_pred = theta + imu_omega[i] * dt
        x_pred = x_est[i-1, 0] + wheel_velocity[i] * np.cos(theta_pred) * dt
        y_pred = x_est[i-1, 1] + wheel_velocity[i] * np.sin(theta_pred) * dt
        
        x_est[i] = [x_pred, y_pred, theta_pred]
        
        # Jacobian F
        F = np.array([
            [1, 0, -wheel_velocity[i] * np.sin(theta_pred) * dt],
            [0, 1,  wheel_velocity[i] * np.cos(theta_pred) * dt],
            [0, 0, 1]
        ])
        
        P = F @ P @ F.T + Q
        
        # ===== Update Step (only if GPS available) =====
        if not np.isnan(gps_x[i]):
            
            z = np.array([gps_x[i], gps_y[i]])
            H = np.array([
                [1, 0, 0],
                [0, 1, 0]
            ])
            
            y_residual = z - H @ x_est[i]
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            
            x_est[i] = x_est[i] + K @ y_residual
            P = (np.eye(3) - K @ H) @ P
    
    return x_est[:,0], x_est[:,1], x_est[:,2]
