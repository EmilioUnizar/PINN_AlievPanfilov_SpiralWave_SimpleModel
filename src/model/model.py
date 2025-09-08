import torch
import pytorch_lightning as pl
import numpy as np
from src.utils.plot_comparison_gif import plot_predictions_vs_true

class PINN(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.criterion = torch.nn.MSELoss()
        self.lr = args.lr
        self.stats = args.stats
        self.factor_ph = args.factor_ph
        self.factor_bc = args.factor_bc
        self.factor_ic = args.factor_ic

        self.scheduler_type = args.scheduler_type

        # Loss containers (maintaining original format)
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.u_gif = []
        self.u_gif_validation = []
        self.counter = 0
        self.test_gif_freq = args.test_gif_freq
        self.warmup_epochs = args.warmup_epochs

        # Define dimensions of introduced data for plotting
        self.Nx, self.Ny, self.Nt = 100, 100, 50
        

    def on_fit_start(self):
        # Collocation points
        # Domain of x ∈ [0, 10], y ∈ [0, 10], t ∈ [0, 150]
        x_c = torch.linspace(0.1, 10, 30)
        y_c = torch.linspace(0.1, 10, 30)
        t_c = torch.linspace(1, 150, 50)
        # Create tridimensional grid
        X_c, Y_c, T_c = torch.meshgrid(x_c, y_c, t_c, indexing='ij')  # indexing='ij' mantiene el orden esperado
        # Flatten and concatenate
        self.collocation_points = torch.stack([X_c.reshape(-1), Y_c.reshape(-1), T_c.reshape(-1)], dim=1) # Shape: (100000, 3)

        # Boundary points
        N_bc_sp = 40
        N_bc_t = 60
        x_bc = torch.linspace(0.1, 10, N_bc_sp)
        y_bc = torch.linspace(0.1, 10, N_bc_sp)
        t_bc = torch.linspace(1, 150, N_bc_t)
        # Complete grid
        X_bc, Y_bc, T_bc = torch.meshgrid(x_bc, y_bc, t_bc, indexing='ij')
        # Select boundary points
        mask = (X_bc == 0.1) | (X_bc == 10) | (Y_bc == 0.1) | (Y_bc == 10)
        Xb = X_bc[mask]
        Yb = Y_bc[mask]
        Tb = T_bc[mask]
        self.boundary_dataset = torch.stack([Xb, Yb, Tb], dim=1)

        # Initial condition points
        # Load data for initial condition
        test_loader = self.trainer.datamodule.test_dataloader()
        for batch in test_loader:
            x_ic, y_ic, t_ic = torch.split(batch['input'], 1, dim=1)
            V_gt_ic, W_gt_ic = torch.split(batch['output'], 1, dim=1)

        x_ic, y_ic, t_ic, V_gt_ic, W_gt_ic = x_ic.to(self.device), y_ic.to(self.device), t_ic.to(self.device), V_gt_ic.to(self.device), W_gt_ic.to(self.device)
        # Initial condition is at t=1 t=43, so we only need to evaluate at t=1 t=43
        idx_init_frente = np.where(np.isclose(t_ic.cpu().numpy(), 1))[0]
        idx_init_rectangle = np.where(np.isclose(t_ic.cpu().numpy(), 43))[0]
        self.x_frente = x_ic[idx_init_frente]
        self.y_frente = y_ic[idx_init_frente]
        self.t_frente = t_ic[idx_init_frente]
        self.V_gt_frente = V_gt_ic[idx_init_frente]
        self.W_gt_frente = W_gt_ic[idx_init_frente]

        self.x_rectangle = x_ic[idx_init_rectangle]
        self.y_rectangle = y_ic[idx_init_rectangle]
        self.t_rectangle = t_ic[idx_init_rectangle]
        self.V_gt_rectangle = V_gt_ic[idx_init_rectangle]
        self.W_gt_rectangle = W_gt_ic[idx_init_rectangle]


    def normalize(self, x, var_type):
        return x

    def denormalize(self, x, var_type):
        return x

    def forward(self, x, y, t):
        """Forward pass with automatic normalization"""
        return self.model(torch.cat([x, y, t], dim=1))

    def physics_loss(self, x, y, t):
        """Compute physics-informed loss""" 
        #Define parameters
        self.D = 0.05
        self.k = 8.0
        self.epsilon = 0.002
        self.mu_1 = 0.2
        self.mu_2 = 0.3
        self.a = 0.01
        self.b = 0.15

        # Ensure inputs require gradients for autograd
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True     
        # Forward pass to get the predicted solution
        out = self.forward(x, y, t)
        V, W = out[:, 0:1], out[:, 1:2]
        # First derivatives
        dv_dx = torch.autograd.grad(V, x, torch.ones_like(V), create_graph=True)[0]
        dv_dy = torch.autograd.grad(V, y, torch.ones_like(V), create_graph=True)[0]
        dv_dt = torch.autograd.grad(V, t, torch.ones_like(V), create_graph=True)[0]
        dw_dt = torch.autograd.grad(W, t, torch.ones_like(W), create_graph=True)[0]
        
        # Second derivatives
        d2v_dx2 = torch.autograd.grad(dv_dx, x, torch.ones_like(dv_dx), create_graph=True)[0]
        d2v_dy2 = torch.autograd.grad(dv_dy, y, torch.ones_like(dv_dy), create_graph=True)[0]

        # Diffusion equation residual
        eq_a = dv_dt -  self.D*(d2v_dx2 + d2v_dy2) + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))

        return (torch.mean(eq_a**2) + torch.mean(eq_b**2))/2  


    def neumann_boundary_loss(self, boundary_points):
        """Compute Neumann boundary condition loss"""
        # Extract points from the boundary dataset
        x, y, t = torch.split(boundary_points.to(self.device), 1, dim=1)

        # Forward pass to get the predicted solution
        out = self.forward(x, y, t)
        V = out[:, 0:1]
        return torch.mean((V)**2)  # Assuming Neumann condition is u = 0 on the boundary
    
    def initial_condition_loss(self):
        """Compute initial condition loss"""
        # Extract points from the initial condition dataset
        out_frente = self.forward(self.x_frente, self.y_frente, self.t_frente)
        V_pred_frente, W_pred_frente = out_frente[:, 0:1], out_frente[:, 1:2]

        out_rectangle = self.forward(self.x_rectangle, self.y_rectangle, self.t_rectangle)
        V_pred_rectangle, W_pred_rectangle = out_rectangle[:, 0:1], out_rectangle[:, 1:2]

        # Initial condition is at t=1
        return (self.criterion(V_pred_frente, self.V_gt_frente) + self.criterion(W_pred_frente, self.W_gt_frente) + self.criterion(V_pred_rectangle, self.V_gt_rectangle) + self.criterion(W_pred_rectangle, self.W_gt_rectangle))/4  # Assuming the initial condition is u(x, y, 1) = 0

    def training_step(self, batch, batch_idx):
        """Training step with data and physics-informed loss"""

        # --- Data loss ---
        x, y, t = torch.split(batch['input'], 1, dim=1)
        V_gt, W_gt = torch.split(batch['output'], 1, dim=1)

        out = self(x, y, t)
        V_pred, W_pred = out[:, 0:1], out[:, 1:2]
        data_loss = self.criterion(V_pred, V_gt) + self.criterion(W_pred, W_gt)

        if self.warmup_epochs > 0 and self.current_epoch < self.warmup_epochs:
            # During warmup, only use data loss
            total_loss = data_loss
        else:
            #After warmup, combine data loss with physics-informed loss
            # --- Physics-informed loss ---
            x_c, y_c, t_c = torch.split(self.collocation_points.to(self.device), 1, dim=1)
            physics_loss = self.physics_loss(x_c, y_c, t_c)

            # --- Neumann boundary condition loss ---
            neumann_loss = self.neumann_boundary_loss(self.boundary_dataset)

            # --- Initial condition loss ---
            initial_condition_loss = self.initial_condition_loss()

            # --- Combine data loss and physics loss ---
            total_loss = data_loss + self.factor_ph * physics_loss + self.factor_bc * neumann_loss + self.factor_ic * initial_condition_loss

            # Check if all losses are finite

            if not torch.isfinite(physics_loss):
                print(f"Physics loss se volvió no finita en la época {self.current_epoch}.")
                self.trainer.should_stop = True
            if not torch.isfinite(neumann_loss):
                print(f"Neumann boundary loss se volvió no finita en la época {self.current_epoch}.")
                self.trainer.should_stop = True
            if not torch.isfinite(initial_condition_loss):
                print(f"Initial condition loss se volvió no finita en la época {self.current_epoch}.")
                self.trainer.should_stop = True
            
            #Logging physics informed losses
            self.log("train_physics_loss", physics_loss, on_step=False, on_epoch=True)
            self.log("train_neumann_loss", neumann_loss, on_step=False, on_epoch=True)
            self.log("train_initial_condition_loss", initial_condition_loss, on_step=False, on_epoch=True)

        
        # Check if all losses are finite
        
        if not torch.isfinite(data_loss):
            print(f"Data loss se volvió no finita en la época {self.current_epoch}.")
            self.trainer.should_stop = True
        if not torch.isfinite(total_loss):
            print(f"Pérdida total se volvió no finita en la época {self.current_epoch}.")
            self.trainer.should_stop = True
        

        # Logging (maintaining original format)
        self.log("train_loss", total_loss, on_step=False, on_epoch=True)
        self.log("train_data_loss", data_loss, on_step=False, on_epoch=True)


        # Log learning rate
        self.log("leaning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False, on_epoch=True)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True):
            x, y, t = torch.split(batch['input'], 1, dim=1)
            V_gt, W_gt = torch.split(batch['output'], 1, dim=1)

            out = self(x, y, t)
            V_pred, W_pred = out[:, 0:1], out[:, 1:2]
            total_loss = self.criterion(V_pred, V_gt) + self.criterion(W_pred, W_gt)
            
            # Logging (maintaining original format)
            self.log("val_loss", total_loss, on_step=False, on_epoch=True)
            
            self.val_losses.append({
                "val_loss": total_loss.item()
            })

            return total_loss
        
    def on_validation_epoch_end(self):
        self.counter += 1 
        if self.counter == self.test_gif_freq:
            self.counter = 0
            test_dataloader = self.trainer.datamodule.test_dataloader()
            for batch in test_dataloader:
                with torch.set_grad_enabled(True):
                    x, y, t = torch.split(batch['input'].to(self.device), 1, dim=1)
                    V_gt, W_gt = torch.split(batch['output'].to(self.device), 1, dim=1)
                    
                    out = self(x, y, t)
                    V_pred, W_pred = out[:, 0:1], out[:, 1:2]
                    data_loss = self.criterion(V_pred, V_gt) + self.criterion(W_pred, W_gt)
                    
                    # Store results for visualization
                    V_net_np = V_pred.detach().cpu().numpy().reshape((self.Nx, self.Ny, self.Nt))
                    V_gt_np = V_gt.detach().cpu().numpy().reshape((self.Nx, self.Ny, self.Nt))
                    W_net_np = W_pred.detach().cpu().numpy().reshape((self.Nx, self.Ny, self.Nt))
                    W_gt_np = W_gt.detach().cpu().numpy().reshape((self.Nx, self.Ny, self.Nt))

                    self.u_gif.append({
                        "V_net": V_net_np,
                        "V_gt": V_gt_np,
                        "W_net": W_net_np,
                        "W_gt": W_gt_np
                    })
                    
                    self.test_losses.append({
                        "data_error": data_loss.item(),
                    })
                
                avg_data_loss = np.mean([loss["data_error"] for loss in self.test_losses])
                    
                self.log("test_data_avg_loss", avg_data_loss)
                self.test_losses.clear()
                
                # Plotting (with denormalization if needed)
                idx = 0
                V_net = self.u_gif[idx]['V_net']
                V_gt = self.u_gif[idx]['V_gt']
                W_net = self.u_gif[idx]['W_net']
                W_gt = self.u_gif[idx]['W_gt']
                
                #values as column vectors
                V_net_l2 = V_net.reshape(-1, 1)
                V_gt_l2 = V_gt.reshape(-1, 1)
                
                # L2 norm error
                rmse_V = np.sqrt(np.mean((V_net_l2 - V_gt_l2) ** 2))
                self.log("rmse_V", rmse_V)

                #if 'u' in self.stats:
                    #u_net = self.denormalize(torch.tensor(u_net), 'u').numpy()
                
             #   plot_predictions_vs_true(
             #       u_pred=V_net,
             #       u_true=V_gt,
             #       x=np.linspace(0.1, 10, 100),
             #       y=np.linspace(0.1, 10, 100),
             #       t=np.linspace(1, 150, 50),
             #       gif_path="comparison_V.gif"
             #   )
             #   plot_predictions_vs_true(
             #       u_pred=W_net,
             #       u_true=W_gt,
             #       x=np.linspace(0.1, 10, 100),
             #       y=np.linspace(0.1, 10, 1),
             #       t=np.linspace(1, 150, 50),
             #       gif_path="comparison_W.gif"
             #   )




    def test_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True):
            x, y, t = torch.split(batch['input'], 1, dim=1)
            V_gt, W_gt = torch.split(batch['output'].to(self.device), 1, dim=1)
            
            out = self(x, y, t)
            V_pred, W_pred = out[:, 0:1], out[:, 1:2]
            
            data_loss = self.criterion(V_pred, V_gt) + self.criterion(W_pred, W_gt)

            # Store results for visualization
            V_net_np = V_pred.detach().cpu().numpy().reshape((self.Nx, self.Ny, self.Nt))
            V_gt_np = V_gt.detach().cpu().numpy().reshape((self.Nx, self.Ny, self.Nt))
            W_net_np = W_pred.detach().cpu().numpy().reshape((self.Nx, self.Ny, self.Nt))
            W_gt_np = W_gt.detach().cpu().numpy().reshape((self.Nx, self.Ny, self.Nt))

            self.u_gif.append({
                "V_net": V_net_np,
                "V_gt": V_gt_np,
                "W_net": W_net_np,
                "W_gt": W_gt_np
            })
            
            self.test_losses.append({
                "data_error": data_loss.item(),
            })

    def on_test_epoch_end(self):
        avg_data_loss = np.mean([loss["data_error"] for loss in self.test_losses])
        self.log("test_data_avg_loss", avg_data_loss)
        self.test_losses.clear()
        
        # Plotting (with denormalization if needed)
        idx = 0
        V_net = self.u_gif[idx]['V_net']
        W_net = self.u_gif[idx]['W_net']
        V_gt = self.u_gif[idx]['V_gt']
        W_gt = self.u_gif[idx]['W_gt']

        #values as column vectors
        V_net_l2 = V_net.reshape(-1, 1)
        W_net_l2 = W_net.reshape(-1, 1)
        V_gt_l2 = V_gt.reshape(-1, 1)
        W_gt_l2 = W_gt.reshape(-1, 1)

        # L2 norm error
        l2_error_V = np.linalg.norm(V_net_l2 - V_gt_l2) / np.linalg.norm(V_gt_l2)
        l2_error_W = np.linalg.norm(W_net_l2 - W_gt_l2) / np.linalg.norm(W_gt_l2)
        global_l2_error = (l2_error_V + l2_error_W) / 2
        self.log("test_l2_error_V", l2_error_V)
        self.log("test_l2_error_W", l2_error_W)
        self.log("test_global_l2_error", global_l2_error)

        # Load training points for visualization
        train_loader = self.trainer.datamodule.train_dataloader()
        xs, ys, ts, Vs, Ws = [], [], [], [], []

        # Reference grid for training data
        x_grid = np.linspace(0.1, 10, self.Nx)
        y_grid = np.linspace(0.1, 10, self.Ny)
        t_grid = np.linspace(1, 150, self.Nt)

        # Empty arrays to store training data
        V_tr = np.full((self.Nx, self.Ny, self.Nt), np.nan)
        W_tr = np.full((self.Nx, self.Ny, self.Nt), np.nan)

        for batch in train_loader:
            x, y, t = torch.split(batch['input'], 1, dim=1)
            V_batch, W_batch = torch.split(batch['output'], 1, dim=1)
            x_np = x.cpu().numpy().flatten()
            y_np = y.cpu().numpy().flatten()
            t_np = t.cpu().numpy().flatten()
            V_np = V_batch.cpu().numpy().flatten()
            W_np = W_batch.cpu().numpy().flatten()
            for xi, yi, ti, vi, wi in zip(x_np, y_np, t_np, V_np, W_np):
                ix = np.argmin(np.abs(x_grid - xi))
                iy = np.argmin(np.abs(y_grid - yi))
                it = np.argmin(np.abs(t_grid - ti))
                V_tr[ix, iy, it] = vi
                W_tr[ix, iy, it] = wi

 #       plot_predictions_vs_true(
 #           u_pred=V_net,
 #           u_true=V_gt,
 #           x=np.linspace(0.1, 10, 100),
 #           y=np.linspace(0.1, 10, 100),
 #           t=np.linspace(1, 150, 50),
 #           u_train_mask=V_tr,
 #           gif_path="comparison_V.gif"
 #       )
 #       plot_predictions_vs_true(
 #           u_pred=W_net,
 #           u_true=W_gt,
 #           x=np.linspace(0.1, 10, 100),
 #           y=np.linspace(0.1, 10, 100),
 #           t=np.linspace(1, 150, 50),
 #           u_train_mask=W_tr,
 #           gif_path="comparison_W.gif"
 #       )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        if self.scheduler_type == 'cosine':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.args.epochs,  # Total number of epochs
                    eta_min=1e-6       # Minimum learning rate
                ),
                'interval': 'epoch',
                'frequency': 1
            }
        elif self.scheduler_type == 'plateau':
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.8,
                    patience=40,
                    threshold=0.0001,
                    min_lr=1e-6
                ),
                'monitor': 'train_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")

        return [optimizer], [scheduler]
