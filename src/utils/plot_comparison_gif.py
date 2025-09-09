import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os

def plot_predictions_vs_true(
    u_pred, u_true, x, y, t, u_train_mask=None,
    gif_path="comparacion_superficie.gif", suavizar=True, sigma=0.0
):
    """
    Crea un GIF 3D con superficies u(x, y) a través del tiempo t, comparando predicción vs. ground truth.

    Args:
        u_pred, u_true: Arrays (N, N, T)
        x, y, t: Arrays 1D
        gif_path: Ruta del gif
        suavizar (bool): Si aplicar suavizado a las superficies
        sigma (float): Parámetro de suavizado gaussiano
    """
    os.makedirs("frames_temp", exist_ok=True)
    filenames = []

    X, Y = np.meshgrid(x, y, indexing='ij')
    umin = min(u_pred.min(), u_true.min())
    umax = max(u_pred.max(), u_true.max())

    plt.ioff()
    
    print('Creating 3D GIF...')

    for i in range(len(t)):
        fig = plt.figure(figsize=(15, 5))

        u_pred_frame = u_pred[:, :, i]
        u_true_frame = u_true[:, :, i]
        error = np.log10(np.abs(u_pred_frame - u_true_frame))

        if u_train_mask is not None:
            mask_frame = u_train_mask[:, :, i]
            train_idx = ~np.isnan(mask_frame)
            x_train = X[train_idx]
            y_train = Y[train_idx]
            z_pred_train = mask_frame[train_idx]
            z_error_train = error[train_idx]

        # --- Predicción ---
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X, Y, u_pred_frame, cmap='viridis', vmin=umin, vmax=umax, alpha=0.8)
        if u_train_mask is not None:
            ax1.scatter(x_train, y_train, z_pred_train, color='red', s=10, label="Train points")
        ax1.set_title(f"Predicción (t={t[i]:.2f})")
        ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("u")
        ax1.set_zlim(0.0, 2.5)
        ax1.view_init(elev=30, azim=-45)

        # --- Ground Truth ---
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X, Y, u_true_frame, cmap='viridis', vmin=umin, vmax=umax, alpha=0.8)
        ax2.set_title("Ground Truth")
        ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("u")
        ax2.set_zlim(0.0, 2.5)
        ax2.view_init(elev=30, azim=-45)

        # --- Error absoluto ---
        ax3 = fig.add_subplot(133, projection='3d')
        surf3 = ax3.plot_surface(X, Y, error, cmap='hot', alpha=0.6)
        if u_train_mask is not None:
            ax3.scatter(x_train, y_train, z_error_train, color='blue', s=10, label="Train points")
        ax3.set_title("Error Absoluto")
        ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("|error|")
        ax3.set_zlim(-0.03, 0.03)
        ax3.view_init(elev=30, azim=-45)

        filename = f"frames_temp/frame_{i:03d}.png"
        plt.tight_layout(pad=3.0)  
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()

    # Crear GIF
    with imageio.get_writer(gif_path, mode='I', duration=0.3, loop=0) as writer:
        for filename in filenames:
            writer.append_data(imageio.imread(filename))

    # Limpiar
    for filename in filenames:
        os.remove(filename)
    os.rmdir("frames_temp")

    print(f"GIF guardado en: {gif_path}")
