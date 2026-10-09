import torch
import numpy as np
import pandas as pd
import anndata as ad
import copy
from tqdm import tqdm
from CytoBridge.tl.methods import neural_ode_step, ODEFunc,MongeMap_simu
from CytoBridge.utils.utils import sample,trace_df_dz,compute_integral,sample_sec
from CytoBridge.tl.losses import calc_ot_loss, calc_mass_loss, calc_score_matching_loss,Density_loss,calc_pinn_loss
from CytoBridge.tl import methods
from CytoBridge.tl.models import DynamicalModel
from CytoBridge.tl.flow_matching import SchrodingerBridgeConditionalFlowMatcher, ConditionalRegularizedUnbalancedFlowMatcher, get_batch_size, compute_uot_plans, get_batch_uot_fm
from CytoBridge.pl.plot import plot_interaction_potential_epoch
from CytoBridge.tl.analysis import simulate_trajectory
from CytoBridge.Map.tl.transport_factory import build_transport_map_from_multi_config

import math
import os
import ot
from torchdiffeq import odeint
from torch.optim.lr_scheduler import StepLR  # Import StepLR scheduler
class TrainingPipeline:
    def __init__(self, model, config, batch_size, device, data, progress_callback=True):  # Added 'data' parameter for initialization
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.optimizer = None
        self.scheduler = None  # Initialize scheduler variable
        self.device = device
        self.model.to(device)
        self.progress_callback = progress_callback  # Callback for progress updates
        # Determine if mass component is used based on model configuration
        self.use_mass = 'growth' in self.config['model']['components']
        # Determine if score component is used based on model configuration
        self.use_score = 'score' in self.config['model']['components']
        # Determine if interaction component is used based on model configuration
        self.use_interaction = 'interaction' in self.config['model']['components']

        # Initialize ODE function (unified gradient calculation entry)
        self.use_multi =self.config['model']["multi"].get('multi_use',False)
        self.use_multisimu =self.config['model']["multi"].get('simu_multi_use',False)

        if (self.use_multi == True) and (self.use_multisimu == False):
            self.tranmap = build_transport_map_from_multi_config(
                self.config['model']["multi"],
                device=device,
            )
            multi_alpha = self.config['model']["multi"].get("multi_alpha",1)
            self.tranmap_simu  = None
        elif (self.use_multi == True) and (self.use_multisimu == True):
            simu_map_params = self.config['model']["multi"].get("simu_map_params", {})
            self.tranmap_simu = MongeMap_simu(
                sigma_hill_A=simu_map_params.get("sigma_hill_A", 0.05),
                sigma_hill_B=simu_map_params.get("sigma_hill_B", 0.05),
                x_center=simu_map_params.get("x_center", 2.0),
                y_center=simu_map_params.get("y_center", 1.5),
            )
            multi_alpha = self.config['model']["multi"].get("multi_alpha",1)
            self.tranmap  = None
        else :
            self.tranmap  = None
            self.tranmap_simu  = None
            multi_alpha = 0
        print(self.tranmap_simu)
        print(self.tranmap)

        # Initialize ODE function (unified gradient calculation entry)
        self.ode_func = ODEFunc(
            model=self.model,
            tranmap=self.tranmap,
            tranmap_simu = self.tranmap_simu,
            sigma=config['training']['defaults'].get('sigma', 0.05),
            multi_alpha=multi_alpha, 
            use_mass=self.use_mass,
            score_use=self.use_score,
            interaction_use=self.use_interaction
        )

        # New: Initialize variables required for train_score_model
        self.logger = self._setup_logger()  # Simple logger implementation
        # Get experiment directory from configuration (default to './results' if not specified)
        self.exp_dir = self.config.get('ckpt_dir', './results')
        os.makedirs(self.exp_dir, exist_ok=True)
        # Construct DataFrame from input data to fit the format required by train_score_model
        self.df = self._prepare_df(data)
        # Get sorted list of unique time points (grouped by 'samples' column)
        self.groups = sorted(self.df.samples.unique())


    def _map_to_secondary(self, x):
        if self.tranmap is not None:
            return self.tranmap.T(x)
        if self.tranmap_simu is not None:
            return self.tranmap_simu.map(x).to(device=x.device, dtype=x.dtype)
        raise ValueError("Secondary OT requires a transport map or simu T map.")


    def _setup_logger(self):
        """Simple logger implementation to replace the original logger"""

        class SimpleLogger:
            @staticmethod
            def info(msg):
                print(f"[INFO] {msg}")

        return SimpleLogger()

    def _prepare_df(self, data):
        """Construct DataFrame from input data to fit the format required by train_score_model
        
        Args:
            data: List of tensors where each element represents samples at a specific time point (shape: n_samples×2)
        
        Returns:
            pd.DataFrame: Combined DataFrame with columns 'x1', 'x2', and 'samples' (time point)
        """
        all_samples = []
        for t_idx, x in enumerate(data):
            x_np = x.cpu().detach().numpy()  # Convert tensor to numpy array
            # Construct DataFrame for current time point: columns = [x1, x2, samples (time point)]
            df_t = pd.DataFrame({
                'x1': x_np[:, 0],
                'x2': x_np[:, 1],
                # Assign current time point to all samples (dtype: float64 for consistency)
                'samples': np.full(x_np.shape[0], t_idx, dtype=np.float64)
            })
            all_samples.append(df_t)
        # Concatenate DataFrames from all time points and reset index
        return pd.concat(all_samples, ignore_index=True)

    # --------------------------
    # Main Modifications: Optimizer and Scheduler Setup
    # --------------------------
    def _setup_stage(self, stage_params):
        lr = stage_params['lr']
        print(f"\n====  {stage_params['name']}  ====")

        # Get flags for score network training from stage parameters
        train_strategy = str(stage_params.get('train_strategy', '')).lower()



        if not train_strategy or train_strategy == 'none':
            train_v = train_g = use_s = use_i = True          # 缺省策略：全训练
        else:
            train_v, train_g, use_s, use_i = 'v' in train_strategy, 'g' in train_strategy, 's' in train_strategy, 'i' in train_strategy

        if stage_params.get('mode') ==  "neural_ode":
            train_s = False
            if train_v and use_s and use_i:
                train_s = True
            self.model.use_growth_in_ode_inter = stage_params.get('use_growth_in_ode_inter', True)
            self.ode_func.use_mass = self.model.use_growth_in_ode_inter
            self.ode_func.score_use = train_strategy is not None and 's' in train_strategy
            self.ode_func.interaction_use = train_strategy is not None and 'i' in train_strategy

        elif stage_params.get('mode') ==  "flow_matching":
            train_s = True
        else:
            raise ValueError(f"Unknown training mode: {stage_params['mode']}")

        self.update_transport = stage_params.get('update_transport', False)
        
        # 根据是否更新传输映射，设置其训练/推理模式
        if self.tranmap is not None:
            map_model = getattr(self.tranmap, "model", None)
            can_toggle = (
                map_model is not None
                and hasattr(map_model, "train")
                and hasattr(map_model, "eval")
            )
            can_update = bool(getattr(self.tranmap, "supports_training", can_toggle))
            if self.update_transport and can_toggle and can_update:
                map_model.train()  # 训练模式：启用梯度
                print(f"  Transportmap: train (update_transport=True)")
            else:
                if can_toggle:
                    map_model.eval()  # 推理模式：禁用梯度
                if self.update_transport and not can_update:
                    print("  warning: update_transport=True, but mapper has no trainable transport parameters")
                print(f"  Transportmap : eval/fixed (update_transport={self.update_transport})")
        else:
            if self.update_transport:
                print(f"  warning: update_transport=True，but tranmap isn't given，will ignore training Transportmap")

        # Collect trainable parameters based on component flags
        params = []

        for name, module in self.model.named_children():
            # print(f"Name: {name}")
            # print(f"Module: {module}")
            # print(f"Module type: {type(module)}")
            # print("Parameters:")
            if (name == 'velocity_net' and train_v) or (name == 'growth_net' and train_g) or  (name == 'score_net' and train_s) or (name == 'interaction_net'  and use_i):
                for p in module.parameters():
                    p.requires_grad = True
                    params.append(p)
                    # print(f"  Parameter shape: {p.shape}")
                    # print(f"  Parameter requires_grad: {p.requires_grad}")
                print("-" * 50)
            else:

                for p in module.parameters():
                    p.requires_grad = False
                    # print(f"  Parameter shape: {p.shape}")
                    # print(f"  Parameter requires_grad: {p.requires_grad}")
        # Initialize Adam optimizer with only trainable parameters
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, params), lr=lr  # 权重衰减，AdamW 核心参数
        )

        # Reset scheduler before setting up new one
        self.scheduler = None
        if 'scheduler_type' in stage_params:
            if stage_params['scheduler_type'] == 'cosine':
                # Use Cosine Annealing scheduler if specified
                cosine_epochs = stage_params.get('cosine_epochs', 1000)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=cosine_epochs, 
                    eta_min=1e-5  # Minimum learning rate
                )
            elif stage_params['scheduler_type'] == 'steplr':
                # Use StepLR scheduler if specified
                self.scheduler = StepLR(
                    optimizer=self.optimizer,
                    step_size=stage_params['scheduler_step_size'],
                    gamma=stage_params['scheduler_gamma']  # Learning rate decay factor
                )
                print(f"  Enabled learning rate scheduler: step_size={stage_params['scheduler_step_size']}, gamma={stage_params['scheduler_gamma']}")
        else:
            print("  No scheduler parameters configured - keeping learning rate constant")

        # Print gradient status (trainable/non-trainable) for each module
        for n, m in self.model.named_children():
            flag = any(p.requires_grad for p in m.parameters())
            print(f"  {n:<15}  grad={flag}")
        # Print shapes of parameters in optimizer
        print("  Optimizer parameters (shapes):", [p.shape for g in self.optimizer.param_groups for p in g['params']])

    def train(self, data, time_points,data_sec_torch=None):
        """Main training loop that executes multiple training stages based on configuration
        
        Args:
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        
        Returns:
            DynamicalModel: Trained model
        """
        # Get training plan and base default parameters from configuration
        training_plan = self.config['training']['plan']
        base_defaults = self.config['training']['defaults']

        # Execute each stage in the training plan
        for stage_config in training_plan:
            # Merge base defaults with stage-specific config (stage config takes priority)
            stage_params = base_defaults.copy()
            stage_params.update(stage_config)
            stage_name = stage_params['name']


            print(f"\n--- Starting Stage: {stage_name} ---")
            print(
                f"  Mode: {stage_params['mode']}, Epochs: {stage_params['epochs']}, Use Score: {stage_params.get('score_use', False)}")
            train_strategy = stage_params.get('train_strategy', None)


            # Setup optimizer, scheduler, and trainable parameters for current stage
            self._setup_stage(stage_params)

            # Execute stage training based on mode
            if stage_params['mode'] == 'neural_ode':
                if data_sec_torch == None:
                    self.run_neural_ode_stage(stage_params, data, time_points)
                else:
                    self.run_neural_ode_stage(stage_params, data, time_points,data_sec_torch)
            elif stage_params['mode'] == 'flow_matching':
                self.run_flow_matching_stage(stage_params, data, time_points)
            else:
                raise ValueError(f"Unknown training mode: {stage_params['mode']}")

        return self.model

    def run_neural_ode_stage(self, stage_params, data, time_points,data_sec_torch=None):
        """Execute training stage using Neural ODE mode
        
        Args:
            stage_params: Dictionary of parameters for current stage (epochs, loss weights, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        """
        epochs = stage_params['epochs']
        # Get model saving strategy (default to 'best' if not specified)
        save_strategy = stage_params.get('save_strategy', 'best')


        # Initialize variables for tracking best model
        best_loss = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        train_strategy = stage_params.get('train_strategy', None)
        train_name=stage_params["name"]

        # Training loop over epochs
        for epoch in range(epochs):
            # Calculate loss for one epoch of Neural ODE training
            loss = self.train_neural_ode_epoch(stage_params, data, time_points, self.ode_func,data_sec_torch)

            if not np.isfinite(loss):
                self.logger.info(
                    f"Neural ODE training stopped at epoch {epoch:3d} due to non-finite loss ({loss}). "
                    "Restoring best model state."
                )
                self.model.load_state_dict(best_state)
                break

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"  Stage '{stage_params['name']}', Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
                # Emit progress callback for Web UI
                if self.progress_callback:
                    progress = (epoch + 1) / epochs
                    self.progress_callback(f"Neural ODE - {stage_params['name']} Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}", progress)

            # if epoch % 10 == 0 and self.use_interaction:
            #     plot_interaction_potential_epoch(self.model,d=1,num_points=40,output_path=self.config["ckpt_dir"]+f"/interfigures/{train_name}_epoch_{epoch}_inter",device="cuda")
            #     if epoch < 15:
            #         print(f"{train_name} plot_interaction_potential_epoch {epoch} has done")

            # if "i" in train_strategy:
            #     if epoch % 10 == 0:
            #         plot_interaction_potential_epoch(self.model,d=1,num_points=21,output_path=self.config["ckpt_dir"]+f"/interfigures/{train_name}_epoch_{epoch}_inter",device="cuda")
            #         print(f"{train_name} plot_interaction_potential_epoch {epoch} has done")
            # Update best model if current loss is lower than previous best
            if loss < best_loss:
                best_loss = loss
                self.logger.info(f"Epoch {epoch:3d} has a lower loss| all_loss {best_loss:.4f}")
                best_state = copy.deepcopy(self.model.state_dict())

        # Determine which model state to save (best or last)
        if save_strategy == 'best':
            save_state = best_state
            save_loss = best_loss
        else:  # 'last' strategy
            save_state = self.model.state_dict()
            # Recalculate loss for last epoch to ensure accuracy
            last_loss = self.train_neural_ode_epoch(stage_params, data, time_points, self.ode_func,data_sec_torch)
            save_loss = last_loss

        # Load saved state (best or last) back to model
        self.model.load_state_dict(save_state)
        # Create checkpoint directory for current stage
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])
        os.makedirs(ckpt_dir, exist_ok=True)
        # Define checkpoint filename based on save strategy
        ckpt_filename = 'best.pth' if save_strategy == 'best_model' else 'last_model.pth'
        torch.save(save_state, os.path.join(ckpt_dir, ckpt_filename))
        print(f"  {save_strategy.capitalize()} model (loss={save_loss:.4f}) saved → {ckpt_dir}/{save_strategy}.pth")

    def train_neural_ode_epoch(self, stage_params, data, time_points,ode_func, data_sec_torch=None):
        """Calculate loss for one epoch of Neural ODE training
        
        Args:
            stage_params: Dictionary of parameters for current stage (loss weights, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
            ode_func: ODEFunc instance for computing ODE updates
        
        Returns:
            float: Average loss over all time intervals
        """
        # Get loss weights and configuration from stage parameters
        lambda_ot = stage_params['lambda_ot']
        lambda_mass = stage_params['lambda_mass']
        lambda_energy = stage_params['lambda_energy']
        
        OT_loss_type = stage_params['OT_loss']
        use_density_loss = stage_params.get('use_density_loss', False)
        use_pinn_loss = stage_params.get('use_pinn_loss', False)

        global_mass = stage_params.get('global_mass', False)
        use_cycle_OT = stage_params.get('use_cycle_OT', False)

        if data_sec_torch != None:
            lambda_ot_sec = stage_params.get('lambda_ot_sec', False)
            lambda_cross_map = stage_params.get('lambda_cross_map', 0.0)
   
        if use_density_loss:
            if 'density_top_k' not in stage_params or 'lambda_density' not in stage_params or 'density_hinge_value' not in stage_params:
                raise ValueError(
                    "When use_density_loss=True, all 'density_top_k','lambda_density' and 'density_hinge_value' "
                    "must be provided in stage_params.(Default recommended ( 5 , 10 and  0.01))" 
                )            
            top_k = stage_params['density_top_k']
            hinge_value = stage_params['density_hinge_value']
            lambda_density = stage_params['lambda_density']
            density_fn = Density_loss(hinge_value)


        # Initialize with sampled data from the first time point
        x0 = sample(data[0], self.batch_size).to(self.device)
        # Initialize log-weights (uniform distribution)
        lnw0 = torch.log(torch.ones(self.batch_size, 1) / self.batch_size).to(self.device)
        # Total number of samples at the first time point
        mass_0 = data[0].shape[0]

        total_loss = 0.0
        # Iterate over all time intervals (from t_{i-1} to t_i)
        for idx in range(1, len(time_points)):
            # Reset gradients before each time interval update
            self.optimizer.zero_grad()

            # Get current time interval and target data
            t0, t1 = time_points[idx - 1], time_points[idx]
            if data_sec_torch != None and lambda_ot_sec!=0 and (data_sec_torch[idx]!= None):
                data_t1, data_t1_sec = sample_sec(data[idx], data_sec_torch[idx], self.batch_size)
                data_t1       = data_t1.to(self.device)
                data_t1_sec   = data_t1_sec.to(self.device)
            else:
                data_t1 = sample(data[idx], self.batch_size).to(self.device)
                
            # Total number of samples at the target time point
            mass_1 = data[idx].shape[0]
            # Calculate relative mass ratio between target and initial time points
            relative_mass = mass_1 / mass_0

            # Perform one Neural ODE step to predict state at t1
            x1, lnw1, e1 = neural_ode_step(ode_func, x0, lnw0, t0, t1, self.device)

            # Calculate individual loss components
            # Calculate individual loss components
            if use_cycle_OT == False and (data_sec_torch == None or lambda_ot_sec == 0):
                loss_ot = calc_ot_loss(x1, data_t1, lnw1, OT_loss_type)

            elif use_cycle_OT == False and (data_sec_torch != None and lambda_ot_sec!=0):
                loss_ot = calc_ot_loss(x1, data_t1, lnw1, OT_loss_type)
                x1_sec = self._map_to_secondary(x1)
                loss_ot_sec = calc_ot_loss(x1_sec, data_t1_sec, lnw1, OT_loss_type)

            elif use_cycle_OT == True and (data_sec_torch != None and lambda_ot_sec!=0):
                x1_sec = self.tranmap.T(x1)
                loss_ot = calc_ot_loss(self.tranmap.T_rev(x1_sec), data_t1_sec, lnw1, OT_loss_type)
                loss_ot_sec = calc_ot_loss(x1_sec, data_t1_sec, lnw1, OT_loss_type)

            else:
                raise ValueError(
                    "When use_cycle_OT=True, sec_adata must be provided" 
                )             
            
            # Calculate mass loss only if mass component is enabled
            loss_mass = calc_mass_loss(x1, data_t1, lnw1, relative_mass, global_mass) if self.use_mass else 0.0
            # Energy loss (average of energy term from ODE step)
            loss_energy = e1.mean()

            # Combine losses with respective weights
            loss = (lambda_ot * loss_ot) + (lambda_mass * loss_mass) + (lambda_energy * loss_energy)

            if use_density_loss:          
                density_loss = density_fn(x1, data_t1, top_k=top_k)
                density_loss = density_loss.to(loss.device)
                loss += lambda_density * density_loss
                # print('density loss')
                # print(density_loss)
            if use_pinn_loss: 
                if 'lambda_pinn'  not in stage_params:
                    raise ValueError(
                        "When use_pinn_loss=True, 'lambda_pinn' must be provided in stage_params.(Default recommended (100))" 
                    )            
                lambda_pinn = stage_params['lambda_pinn'] 

                loss_pinn = calc_pinn_loss(self, t1, data_t1,sigma=stage_params['sigma'], use_mass=self.use_mass,trace_df_dz=trace_df_dz,device=self.device)
                # print("loss_pinn",loss_pinn)
                # print("loss",loss)
                loss += lambda_pinn * loss_pinn

            print(f"OT Loss: {loss_ot:.4f} (λ={lambda_ot}), Mass Loss: {loss_mass:.4f} (λ={lambda_mass}), Energy Loss: {loss_energy:.4f} (λ={lambda_energy}), Density Loss: {density_loss:.4f} (λ={lambda_density})" if use_density_loss else f"OT Loss: {loss_ot:.4f} (λ={lambda_ot}), Mass Loss: {loss_mass:.4f} (λ={lambda_mass}), Energy Loss: {loss_energy:.4f} (λ={lambda_energy})", end="\n")
            if use_pinn_loss:
                print(f", PINN Loss: {loss_pinn:.4f} (λ={lambda_pinn})")
            if data_sec_torch != None and lambda_ot_sec!=0:
                print(f", loss_ot_sec Loss: {loss_ot_sec:.4f} (λ={lambda_ot_sec})")
                loss_ot_sec = lambda_ot_sec * loss_ot_sec
                loss += loss_ot_sec
            if self.update_transport and (data_sec_torch is not None) and (lambda_cross_map > 0) :
                mapped_data = self._map_to_secondary(data_t1)
                cross_loss = torch.nn.functional.mse_loss(mapped_data, data_t1_sec)
                loss = loss + lambda_cross_map * cross_loss
                print(f"  cross_loss: {cross_loss.item():.4f} (λ={lambda_cross_map})")
            
            loss.backward()
            self.optimizer.step()

            # Update initial state for next time interval (detach to avoid gradient accumulation)
            x0 = x1.clone().detach()
            lnw0 = lnw1.clone().detach()

            # Accumulate total loss over all time intervals
            total_loss += loss.item()

        # Return average loss per time interval
        return total_loss / (len(time_points) - 1)


    def run_flow_matching_stage(self, stage_params, data, time_points):
        """Execute training stage using Flow Matching mode
        
        Args:
            stage_params: Dictionary of parameters for current stage (epochs, sigma, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        """
        # Create checkpoint directory for current stage
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])
        os.makedirs(ckpt_dir, exist_ok=True)

        # Convert time points to tensor (device-compatible)
        time = torch.tensor(time_points, device=self.device, dtype=torch.float32)
        # Get sigma parameter for Flow Matching
        sigma = stage_params['sigma']
        # Get alpha regularization parameter (default to 1.0 if not specified)
        alpha_regm = stage_params.get('alpha_regm', 1.0)
        print("alpha_regm :", alpha_regm)
        self.sigma = sigma
        # Convert data to list of numpy arrays (required for compute_uot_plans)
        X = [data[i].float().cpu().detach().numpy() for i in range(len(time_points))]
        
        # Get flags for training different network components
        train_strategy = str(stage_params.get('train_strategy', 's')).lower()
        regress_v, regress_g, regress_score = 'v' in train_strategy, 'g' in train_strategy, 's' in train_strategy
        
        if regress_g or regress_v :
            uot_plans, sampling_info = compute_uot_plans(X, time_points,use_mini_batch_uot=True, chunk_size=1000, alpha_regm= alpha_regm ,reg_strategy="max_over_time", device=self.device)
        else :
            uot_plans, sampling_info = compute_uot_plans(X, time_points,use_mini_batch_uot=True, chunk_size=2000,reg_strategy='per_time', device=self.device)

        # Initialize Conditional Regularized Unbalanced Flow Matcher
        FM = ConditionalRegularizedUnbalancedFlowMatcher(sigma=sigma)
        # Get model saving strategy (default to 'best' if not specified)
        save_strategy = stage_params.get('save_strategy', 'best')
        # Initialize variables for tracking best model
        best_loss = float('inf')
        best_state_dict = None
        
        # Get batch size from stage parameters
        batch_size = stage_params['batch_size']



        # Training loop over epochs (with tqdm progress bar)
        total_epochs = stage_params['epochs']
        for epoch in tqdm(range(total_epochs), desc='Flow matching'):
            # Calculate loss for one epoch of Flow Matching training
            loss, penalty = self.train_flow_matching_epoch(
                FM, X, time,
                self.optimizer,
                stage_params['flow_matching']['lambda_penalty'],
                batch_size,
                uot_plans,
                sampling_info,
                regress_v, regress_g, regress_score,
            )

            # Stop training if loss becomes NaN (numerical instability)
            if torch.isnan(loss):
                self.logger.info("Training stopped due to NaN loss")
                # Load best model state before NaN occurred
                self.model.load_state_dict(best_state_dict)
                break

            # Update best model if current loss is lower than previous best
            if loss < best_loss:
                best_loss = loss
                best_state_dict = self.model.state_dict().copy()

            # Emit progress callback for Web UI every 10 epochs
            if epoch % 10 == 0 and self.progress_callback:
                progress = (epoch + 1) / total_epochs
                loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
                self.progress_callback(f"Flow Matching Epoch {epoch + 1}/{total_epochs}, Loss: {loss_val:.4f}", progress)

            # Combine loss and penalty for backpropagation
            total_loss = loss + penalty
            # print("score_loss",loss)
            # print("penalty",penalty)

            total_loss.backward()
            # Update optimizer
            self.optimizer.step()
            # Update scheduler if initialized
            if self.scheduler is not None:
                self.scheduler.step()


        # Determine which model state to save (best or last)
        if save_strategy == 'best':
            save_state = best_state_dict
            save_loss = best_loss
        else:  # 'last' strategy
            save_state = self.model.state_dict()
            save_loss = loss.item() + penalty.item()

        # Load saved state (best or last) back to model
        self.model.load_state_dict(save_state)
        # Define checkpoint filename based on save strategy
        ckpt_filename = 'best_model.pth' if save_strategy == 'best' else 'last_model.pth'
        torch.save(save_state, os.path.join(ckpt_dir, ckpt_filename))
        print(f"  {save_strategy.capitalize()} model (loss={save_loss:.4f}) "
              f"saved → {ckpt_dir}/{save_strategy}_model.pth")

    def train_flow_matching_epoch(self, FM, X, time,
                                  optimizer, lambda_pen, batch_size, uot_plans, sampling_info, regress_v, regress_g, regress_score):
        """Calculate loss for one epoch of Flow Matching training
        
        Args:
            FM: ConditionalRegularizedUnbalancedFlowMatcher instance
            X: List of numpy arrays where each element represents samples at a specific time point
            time: Tensor of time points (device-compatible)
            optimizer: Torch optimizer instance
            lambda_pen: Penalty weight for score network training
            batch_size: Batch size for sampling
            uot_plans: Precomputed UOT plans for sampling
            sampling_info: Additional sampling information from compute_uot_plans
            regress_v: Flag to train velocity network (v)
            regress_g: Flag to train growth network (g)
            regress_score: Flag to train score network
        
        Returns:
            tuple: (total_loss, penalty) where both are torch tensors
        """
        # Reset gradients before each batch
        optimizer.zero_grad()
        # Sample batch data for Flow Matching (time, positions, velocities, growth values, weights, noise)
        t, xt, ut, gt_samp, weights, eps = get_batch_uot_fm(FM, X, time, batch_size, uot_plans, sampling_info, device=self.device)
        # Reshape time tensor to (batch_size, 1) for concatenation with position data
        t = torch.unsqueeze(t, 1).to(self.device)

        # Compute lambda(t) (time-dependent weighting factor for score network)
        t_floor = torch.zeros_like(t)
        t_ceil = torch.zeros_like(t)
        # Determine time interval bounds (t_floor and t_ceil) for each sample in the batch
        for j in range(len(time) - 1):
            mask = (t >= time[j]) & (t < time[j + 1])
            t_floor[mask] = time[j]
            t_ceil[mask] = time[j + 1]
        # Calculate normalized time within interval and compute lambda(t)
        lambda_t = FM.compute_lambda((t - t_floor) / (t_ceil - t_floor))

        # Enable gradient computation for position data (required for score calculation via autograd)
        xt = xt.requires_grad_(True)
        # Get references to model components
        v_net = self.model.velocity_net
        g_net = self.model.growth_net
        score_net = self.model.score_net
        # Concatenate position and time data for network input (shape: batch_size × (2 + 1) = batch_size × 3)
        net_input = torch.cat([xt, t], dim=1)

        # Initialize loss and penalty
        loss = 0.0
        penalty = 0.0
        # Train score network if enabled
        if regress_score:
            # Predict score potential (value_st) from score network
            value_st = score_net(net_input)
            # Compute score via automatic differentiation (gradient of value_st w.r.t. xt)
            st = torch.autograd.grad(
                outputs=value_st,
                inputs=xt,
                grad_outputs=torch.ones_like(value_st),
                create_graph=True  # Required for second-order gradients (if needed)
            )[0]
            # Calculate weighted MSE loss for score network
            score_loss = torch.mean(weights * ((lambda_t[:, None] * st + eps) ** 2))
            # Handle NaN loss (set to 0 to avoid training instability)
            if torch.isnan(score_loss):
                score_loss = 0.0
            loss += score_loss
            # Add penalty term to regularize score potential (prevents exploding values)
            penalty += lambda_pen * torch.max(torch.relu(value_st))
        
        # Train velocity network (v) if enabled
        if regress_v:
            # Predict velocity from velocity network
            v_predict = v_net(net_input)
            # Add weighted MSE loss between predicted and target velocities
            loss += torch.mean(weights * (v_predict - ut) ** 2)
        
        # Train growth network (g) if enabled
        if regress_g:
            # Predict growth values from growth network
            g_predict = g_net(net_input)
            # Add weighted MSE loss between predicted and target growth values (scaled by 1000 for better convergence)
            loss += 1000 * torch.mean(weights * (g_predict - gt_samp) ** 2)

        return torch.as_tensor(loss, device=self.device), torch.as_tensor(penalty, device=self.device)
    def evaluate(self,adata, data, time_points,val=None):
        """Evaluate trained model using Wasserstein-1 distance and Total Mass Variation (TMV)
        
        Args:
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        
        Returns:
            list: List of Wasserstein-1 distances for each time point (excluding initial time)
        """
        print(f"\n--- Starting Evaluation ---")
        device = self.device
        # Get initial time point data (t=0)
        if isinstance(val, (int, float)):   # 纯 Python 数字
            n = int(val)
            T = len(data)
            N0 = data[0].shape[0]
            n0 = min(n, N0)
            ratio = n0 / N0 if N0 > 0 else 1.0
            generator = torch.Generator()
            generator.manual_seed(42)
            idx0 = torch.randperm(N0, generator=generator)[:n0]
            x0 = data[0][idx0].to(device)
            new_data = [None] * T
            new_data[0] = data[0][idx0]
            for t in range(1, T):
                Nt = data[t].shape[0]
                nt = min(max(int(round(Nt * ratio)), 1), Nt)
                gen_t = torch.Generator()
                gen_t.manual_seed(42 + t)
                idx_t = torch.randperm(Nt, generator=gen_t)[:nt]
                new_data[t] = data[t][idx_t]
            data = new_data
        else:
            x0 = data[0].to(device)
        # Freeze model parameters during evaluation (disable gradient computation)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get sigma parameter (use stored value or default to 0.05 if not available)
        sigma = 0

        # Simulate trajectory using the trained model
        point, weight = simulate_trajectory(
            adata,
            self.model,
            x0,
            sigma,           
            time_points,
            dt=0.1,  # Time step for ODE simulation
            device=x0.device
        )

        # Calculate Wasserstein-1 distance for each time point (excluding initial time)
        wasserstein_scores = []
        tmv_scores = []
        for idx in range(1, len(time_points)):
            t0, t1 = time_points[0], time_points[idx]
            # Get target data at current time point (convert to numpy for OT computation)
            data_t1 = data[idx].detach().cpu().numpy()
            # Get predicted positions and weights from simulated trajectory
            x1 = point[idx]
            m1 = weight[idx]

            # Calculate Total Mass Variation (TMV) between predicted and true mass
            tmv = np.abs(m1.sum() - data[idx].shape[0] / data[0].shape[0])
            # Normalize predicted weights to sum to 1 (required for OT)
            m1 = m1 / m1.sum()

            # Create uniform weights for target data (sum to 1)
            m2 = np.ones(data_t1.shape[0]) / data_t1.shape[0]
            # Compute Euclidean distance matrix between target and predicted points
            cost_matrix = ot.dist(data_t1, x1, metric='euclidean')

            # Calculate Wasserstein-1 distance using Earth Mover's Distance (EMD)
            w1 = ot.emd2(
                m2,
                m1.reshape(-1),  # Reshape to 1D array (required by ot.emd2)
                cost_matrix,
                numItermax=1e7  # Increase max iterations for convergence
            )

            # Store results and print progress
            wasserstein_scores.append(w1)
            tmv_scores.append(tmv)
            print(f"  Time Point {t1}: Wasserstein-1 Distance = {w1:.4f}")
            print(f"  Time Point {t1}: TMV = {tmv:.4f}")
        
        return {
            'w1_scores': wasserstein_scores,
            'tmv_scores': tmv_scores
        }

    def evaluate_merge(self, adata, data, time_points,dim=50 , val=None):
        print(f"\n--- Starting Evaluation (merge mode) ---")
        device = self.device

        # ----- 初始采样 -----
        if isinstance(val, (int, float)):
            n = int(val)
            torch.manual_seed(42)               # 固定种子，42 可换成任意整数
            idx = torch.randperm(len(data[0]))[:n] # 每次都会得到同一组索引
            x0 = data[0][idx].to(device)
            T = len(data)
            N = data[0].shape[0]
            ratio = n / N
            generator = torch.Generator()
            generator.manual_seed(42)
            idx = torch.randperm(N, generator=generator)[:n]  
            new_data = [None] * T
            for t in range(0, T):
                new_data[t] = data[t][idx]
            data=new_data
        else:
            x0 = data[0].to(device)

        for p in self.model.parameters():
            p.requires_grad = False

        sigma = 0
        point, weight = simulate_trajectory(
            adata, self.model, x0, sigma, time_points, dt=0.1, device=device
        )

        results = []
        for idx, t in enumerate(time_points):
            # 真实数据
            data_t = data[idx].detach().cpu().numpy()      # (N, 100)
            N, D = data_t.shape
            half = dim

            data_front = data_t[:, :half]
            data_back  = data_t[:, half:]
            print(data_front.shape,data_back.shape)

            # 预测数据
            x1 = point[idx]            # (N, 100)
            m1 = weight[idx]           # (N,)
            m1 = m1 / m1.sum()         # 归一化
            x1_front = x1[:, :half]
            x1_back  = x1[:, half:]


            # 前半
            m2_front = np.ones(data_front.shape[0]) / data_front.shape[0]
            C_front  = ot.dist(data_front, x1_front, metric='euclidean')
            w1_front = ot.emd2(m2_front, m1, C_front, numItermax=int(1e7))
            tmv_front = np.abs(m1.sum() - data_front.shape[0] / data[0].shape[0])

            m2_back = np.ones(data_back.shape[0]) / data_back.shape[0]
            C_back  = ot.dist(data_back, x1_back, metric='euclidean')
            w1_back = ot.emd2(m2_back, m1, C_back, numItermax=int(1e7))
            tmv_back = np.abs(m1.sum() - data_back.shape[0] / data[0].shape[0])

            # 保证是标量
            w1_front, tmv_front = float(w1_front[0]), float(tmv_front)
            w1_back,  tmv_back  = float(w1_back[0]),  float(tmv_back)

            results.append((w1_front, tmv_front, w1_back, tmv_back))
            print(f"  Time {t}:  front(W1={w1_front:.4f}, TMV={tmv_front:.4f}) | "
                f"back(W1={w1_back:.4f}, TMV={tmv_back:.4f})")

        return results

    def evaluate_sec_fir_sec1(self,adata,tranmap_fir_sec, data_torch_sec_fir,data_torch_sec, time_points,val=None):
        #second data map to first space,calculate trajecories in first space, at the end simulated  trajecories map to second space, calculate Wasserstein-1 distance and Total Mass Variation in second space
        """Evaluate trained model using Wasserstein-1 distance and Total Mass Variation (TMV)
        
        Args:
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        
        Returns:
            list: List of Wasserstein-1 distances for each time point (excluding initial time)
        """
        print(f"\n--- Starting Evaluation ---")
        device = self.device
        # Get initial time point data (t=0)
        if isinstance(val, (int, float)):   # 纯 Python 数字
            n = int(val)
            T = len(data_torch_sec_fir)
            N0 = data_torch_sec_fir[0].shape[0]
            n0 = min(n, N0)
            ratio = n0 / N0 if N0 > 0 else 1.0
            generator = torch.Generator()
            generator.manual_seed(42)
            idx0 = torch.randperm(N0, generator=generator)[:n0]
            x0 = data_torch_sec_fir[0][idx0].to(device)
            print( data_torch_sec_fir[0].shape)
            new_data = [None] * T
            new_data[0] = data_torch_sec_fir[0][idx0]
            for t in range(1, T):
                Nt = data_torch_sec_fir[t].shape[0]
                nt = min(max(int(round(Nt * ratio)), 1), Nt)
                gen_t = torch.Generator()
                gen_t.manual_seed(42 + t)
                idx_t = torch.randperm(Nt, generator=gen_t)[:nt]
                new_data[t] = data_torch_sec_fir[t][idx_t]
            data_torch_sec_fir = new_data
        else:
            # 你原来的随机采样逻辑
            x0 = data_torch_sec_fir[0].to(device)
        # Freeze model parameters during evaluation (disable gradient computation)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get sigma parameter (use stored value or default to 0.05 if not available)
        sigma = getattr(self, 'sigma', None) or 0.05
        sigma = 0

        # Simulate trajectory using the trained model
        point, weight = simulate_trajectory(
            adata,
            self.model,
            x0,
            sigma,           
            time_points,
            dt=0.1,  # Time step for ODE simulation
            device=x0.device
        )

        # Calculate Wasserstein-1 distance for each time point (excluding initial time)
        wasserstein_scores = []
        for idx in range(0, len(time_points)):
            t0, t1 = time_points[0], time_points[idx]
            # Get target data at current time point (convert to numpy for OT computation)
            data_t1 = data_torch_sec[idx].detach().cpu().numpy()
            # Get predicted positions and weights from simulated trajectory
            x1 = point[idx]
            x1 = tranmap_fir_sec.T(x1).detach().cpu().numpy()
            m1 = weight[idx]
            
            # Calculate Total Mass Variation (TMV) between predicted and true mass
            tmv = np.abs(m1.sum() - data_torch_sec[idx].shape[0] / data_torch_sec[0].shape[0])
            # Normalize predicted weights to sum to 1 (required for OT)
            m1 = m1 / m1.sum()

            # Create uniform weights for target data_torch_sec (sum to 1)
            m2 = np.ones(data_t1.shape[0]) / data_t1.shape[0]
            # Compute Euclidean distance matrix between target and predicted points
            cost_matrix = ot.dist(data_t1, x1, metric='euclidean')

            # Calculate Wasserstein-1 distance using Earth Mover's Distance (EMD)
            w1 = ot.emd2(
                m2,
                m1.reshape(-1),  # Reshape to 1D array (required by ot.emd2)
                cost_matrix,
                numItermax=1e7  # Increase max iterations for convergence
            )

            # Store results and print progress
            wasserstein_scores.append(w1)
            print(f"  Time Point {t1}: Wasserstein-1 Distance = {w1:.4f}")
            print(f"  Time Point {t1}: TMV = {tmv:.4f}")
        
        return wasserstein_scores

    def evaluate_map(self,data, sub_data):
        from sklearn.metrics import mean_squared_error
        from scipy.spatial.distance import cosine


        data_t = torch.as_tensor(data, dtype=torch.float32, device=self.device)
        output_data = self._map_to_secondary(data_t).detach().cpu().numpy()
        if self.tranmap is not None:
            print("mode:",self.tranmap.mode)
        elif self.tranmap_simu is not None:
            print("mode: simu_T")
        """计算量化指标"""
        mse = mean_squared_error(sub_data, output_data)
        nmse = mse / np.var(sub_data) if np.var(sub_data) > 1e-8 else mse
        cos_sim = np.mean([1 - cosine(o, r) for o, r in zip(sub_data, output_data)])
        mae = np.mean(np.abs(sub_data - output_data))
        rel_error = mae / (np.mean(np.abs(sub_data)) + 1e-8)
        # 打印
        print(f"    MSE: {mse:.6f} | NMSE: {nmse:.6f}")
        print(f"    余弦相似度: {cos_sim:.4f} (理想=1)")
        print(f"    MAE: {mae:.6f} | 相对误差: {rel_error:.4f} (理想=0)")
