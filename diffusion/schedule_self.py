import torch
import math 
from tqdm import tqdm

class Diffusion_Scheduler:
    #Noise scheduling will be done using this class | Written By Ayush Raina
    
    def __init__(self, number_of_timesteps=1000, beta_not=0.0001, beta_T=0.02, beta_schedule='linear'):
        # Get all noise scheduling parameters
        # Args: t, b_0, b_T
        
        # TimeSteps
        self.numberOfTimesteps = number_of_timesteps
        
        # Variance Schedule - BETA SCHEDULES (not timestep spacing!)
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_not, beta_T, number_of_timesteps)
        elif beta_schedule == 'cosine':
            self.betas = self.GetCosineBetaSchedule(self.numberOfTimesteps)
        elif beta_schedule == 'quadratic':
            self.betas = self.GetQuadraticBetaSchedule(self.numberOfTimesteps, beta_not, beta_T)
        elif beta_schedule == 'sigmoid':
            self.betas = self.GetSigmoidBetaSchedule(self.numberOfTimesteps, beta_not, beta_T)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Gaussian Alphas
        self.alpha_t = 1 - self.betas
        self.alpha_t_bar = torch.cumprod(self.alpha_t, dim=0)
        self.sqrt_alpha_t_bar = torch.sqrt(self.alpha_t_bar)
        self.sqrt_one_minus_alpha_t_bar = torch.sqrt(1 - self.alpha_t_bar)
        
    def GetCosineBetaSchedule(self, T, s=0.008):
        """
        Cosine schedule for BETA values as proposed in Nichol & Dhariwal 2021.
        This creates a cosine schedule for the NOISE SCHEDULE (betas), not timesteps.
        """
        steps = torch.arange(T + 1, dtype=torch.float64) / T
        alpha_bars = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]  # normalize to 1 at t=0

        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = torch.clamp(betas, min=1e-5, max=0.999)  # clip values for stability

        return betas.float()
    
    def GetQuadraticBetaSchedule(self, T, beta_start=0.0001, beta_end=0.02):
        """
        Quadratic schedule for BETA values.
        More gradual increase in noise compared to linear.
        """
        t = torch.linspace(0, 1, T)
        betas = beta_start + (beta_end - beta_start) * (t ** 2)
        return betas.float()
    
    def GetSigmoidBetaSchedule(self, T, beta_start=0.0001, beta_end=0.02):
        """
        Sigmoid schedule for BETA values.
        S-shaped curve - slow start, rapid middle, slow end.
        """
        t = torch.linspace(-6, 6, T)  # Range for sigmoid
        sigmoid = torch.sigmoid(t)
        # Normalize to [0, 1] and scale to beta range
        sigmoid_norm = (sigmoid - sigmoid.min()) / (sigmoid.max() - sigmoid.min())
        betas = beta_start + (beta_end - beta_start) * sigmoid_norm
        return betas.float()
    
    def create_timesteps_for_DDIM_accelerated_sampling(self, number_of_inference_timesteps, spacing="linear"):
        """
        Create timesteps with different SPACING strategies.
        This is about which timesteps to sample, NOT the beta schedule.
        """
        if spacing == "linear":
            timesteps = torch.linspace(self.numberOfTimesteps-1, 0, number_of_inference_timesteps).long()
        
        elif spacing == "quadratic":
            # More steps at the beginning (high noise levels)
            t_normalized = torch.linspace(0, 1, number_of_inference_timesteps)
            t_quadratic = t_normalized ** 2
            timesteps = ((1 - t_quadratic) * (self.numberOfTimesteps - 1)).long()
        
        elif spacing == "sqrt":
            # More steps at the end (low noise levels)
            t_normalized = torch.linspace(0, 1, number_of_inference_timesteps)
            t_sqrt = torch.sqrt(t_normalized)
            timesteps = ((1 - t_sqrt) * (self.numberOfTimesteps - 1)).long()
            
        elif spacing == "cosine":
            # Cosine spacing for timesteps (often works well)
            steps = torch.arange(number_of_inference_timesteps)
            timesteps = ((torch.cos(steps * math.pi / (2 * number_of_inference_timesteps)) ** 2) * (self.numberOfTimesteps - 1)).long()
        
        elif spacing == "exponential":
            # Exponential spacing - more steps at high noise
            t_normalized = torch.linspace(0, 1, number_of_inference_timesteps)
            t_exp = torch.exp(t_normalized * 3) - 1  # Adjust the 3 to control curve
            t_exp = t_exp / t_exp.max()  # Normalize
            timesteps = ((1 - t_exp) * (self.numberOfTimesteps - 1)).long()
            
        return timesteps
    
    def extract_value_at_timestep_t(self, values, timestep_t, broadcast_shape):
        batch_size = timestep_t.shape[0]
        output = values.to(timestep_t.device).gather(0, timestep_t)
        return output.reshape(batch_size, *([1] * (len(broadcast_shape) - 1)))
    
    def addNoiseKernel(self, x_0, timestep_t, noise=None):
        # This function will return the noisy image at timestep t
        if noise is None:
            noise = torch.randn_like(x_0)
            
        a = self.extract_value_at_timestep_t(self.sqrt_alpha_t_bar, timestep_t, x_0.shape)
        b = self.extract_value_at_timestep_t(self.sqrt_one_minus_alpha_t_bar, timestep_t, x_0.shape)
        
        x_t = a*x_0 + b*noise
        return x_t, noise
    
    def predict_x0_from_noise(self, x_t, timestep_t, predicted_epsilon_t):
        # Variant - Compute x_0|t from x_t and predicted noise
        sqrt_alpha_t_bar = self.extract_value_at_timestep_t(self.sqrt_alpha_t_bar, timestep_t, x_t.shape)
        sqrt_one_minus_alpha_t_bar = self.extract_value_at_timestep_t(self.sqrt_one_minus_alpha_t_bar, timestep_t, x_t.shape)
        
        # Calculate Predicted x_0
        x_0_given_t = (x_t - sqrt_one_minus_alpha_t_bar * predicted_epsilon_t) / sqrt_alpha_t_bar
        
        # Clamping 
        x_0_given_t = torch.clamp(x_0_given_t, -1.0, 1.0)
        
        return x_0_given_t
        
    def calculate_mean_for_previous_timestep_DDPM_Version(self, x_t, timestep_t, x_0_given_t):
        """
        Compute the mean of the distribution q(x_{t-1} | x_t, x_0)
        
        Args:
            x_t: Current noisy image at timestep t
            t: Current timestep indices
            pred_x0: Predicted x_0 from x_t and predicted noise
            
        Returns:
            Mean of the distribution for x_{t-1}
        """
        # Extract needed values
        
        alpha_t = self.extract_value_at_timestep_t(self.alpha_t, timestep_t, x_t.shape)
        alpha_t_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, timestep_t, x_t.shape)
        beta_t = self.extract_value_at_timestep_t(self.betas, timestep_t, x_t.shape)
        
        # Handle t=0 as a special case
        if timestep_t[0] == 0:
            return x_0_given_t
        
        # For t>0, we need to compute the posterior mean
        # Get alpha_{t-1}_bar
        t_minus_1 = torch.maximum(timestep_t - 1, torch.tensor([0], device=timestep_t.device))
        alpha_t_minus_1_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, t_minus_1, x_t.shape)
        
        # Compute the coefficients for the posterior mean
        coef1 = (torch.sqrt(alpha_t_minus_1_bar) * beta_t) / (1 - alpha_t_bar)
        coef2 = (torch.sqrt(alpha_t) * (1 - alpha_t_minus_1_bar)) / (1 - alpha_t_bar)
        
        # Compute the posterior mean
        mean_for_normal_x_t_minus_1 = coef1 * x_0_given_t + coef2 * x_t
        
        return mean_for_normal_x_t_minus_1
    
    def calculate_mean_for_previous_timestep_DDIM_Version1(self, x_t, timestep_t, x_0_given_t, sigma_t):
        
        t_minus_1 = torch.maximum(timestep_t-1, torch.tensor([0], device=timestep_t.device))
        alpha_t_minus_1_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, t_minus_1, x_t.shape)
        
        sqrt_alpha_t_bar = self.extract_value_at_timestep_t(self.sqrt_alpha_t_bar, timestep_t, x_t.shape)
        sqrt_one_minus_alpha_t_bar = self.extract_value_at_timestep_t(self.sqrt_one_minus_alpha_t_bar, timestep_t, x_t.shape)
        
        coef_one = torch.sqrt(alpha_t_minus_1_bar)
        coef_two = torch.sqrt(1 - alpha_t_minus_1_bar - sigma_t**2) / sqrt_one_minus_alpha_t_bar
        
        mean_for_normal_x_t_minus_1_DDIM_version = coef_one * x_0_given_t + coef_two * (x_t - sqrt_alpha_t_bar * x_0_given_t)
        return mean_for_normal_x_t_minus_1_DDIM_version
    
    def calculate_mean_for_previous_timestep_DDIM_Version2(self, x_t, timestep_t, predicted_noise, sigma_t):
        
        t_minus_1 = torch.maximum(timestep_t - 1, torch.tensor([0], device = timestep_t.device))
        alpha_t_minus_1_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, t_minus_1, x_t.shape)
        alpha_t_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, timestep_t, x_t.shape)
        alpha_t = self.extract_value_at_timestep_t(self.alpha_t, timestep_t, x_t.shape)
        
        sqrt_alpha_t_minus_1_bar = torch.sqrt(alpha_t_minus_1_bar)
        sqrt_alpha_t_bar = torch.sqrt(alpha_t_bar)
        coef_one = torch.sqrt(1-alpha_t_bar)
        term1 = sqrt_alpha_t_minus_1_bar * ((x_t - coef_one * predicted_noise) / sqrt_alpha_t_bar)
        
        coef_two = torch.sqrt(1 - alpha_t_minus_1_bar - sigma_t**2) * predicted_noise
        
        return term1 + coef_two
        
    
    def calculate_mean_for_previous_timestep_accelerated_DDIM_Version(self, x_t, timestep_t, timestep_t_minus_1, x_0_given_t, sigma_t):
        
        alpha_t_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, timestep_t, x_t.shape)
        alpha_t_minus_1_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, timestep_t_minus_1, x_t.shape)
        
        sqrt_alpha_t_bar = torch.sqrt(alpha_t_bar)
        sqrt_alpha_t_minus_1_bar = torch.sqrt(alpha_t_minus_1_bar)
        sqrt_1_minus_alpha_t_bar = torch.sqrt(1-alpha_t_bar)
        
        coef_one = sqrt_alpha_t_minus_1_bar
        coef_two = torch.sqrt(1 - alpha_t_minus_1_bar - sigma_t**2) / sqrt_1_minus_alpha_t_bar
        
        mean_accelerated_ddim = coef_one * x_0_given_t + coef_two * (x_t - sqrt_alpha_t_bar * x_0_given_t)
        return mean_accelerated_ddim
        
    def calculate_variance_for_previous_timestep_DDPM_Version(self, timestep_t, precictedVariance = None):
        
        """
        Compute the variance of the distribution q(x_{t-1} | x_t, x_0)
        
        Args:
            t: Current timestep indices
            predicted_variance: Optional predicted variance from model
            
        Returns:
            Variance of the distribution for x_{t-1}
        """
        # Extract beta_t
        batch_size = timestep_t.shape[0]
        beta_t = self.extract_value_at_timestep_t(self.betas, timestep_t, [batch_size, 1, 1, 1])
        
        # Handle t=0 as a special case
        if timestep_t[0] == 0:
            return torch.zeros_like(beta_t)
        
        # For t>0, we use fixed variance (DDPM paper)
        # Get alpha_{t-1}_bar
        t_minus_1 = torch.maximum(timestep_t - 1, torch.tensor([0], device=timestep_t.device))
        alpha_t_minus_1_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, t_minus_1, [batch_size, 1, 1, 1])
        alpha_t_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, timestep_t, [batch_size, 1, 1, 1])
        
        # Fixed variance as in DDPM paper
        variance = (1 - alpha_t_minus_1_bar) / (1 - alpha_t_bar) * beta_t
        
        return variance
    
    def calculate_variance_for_previous_timestep_DDIM_Version(self, timestep_t, eta=1):
        """
        Calculate DDIM variance correctly
        """
        batch_size = timestep_t.shape[0]
        
        # Handle t=0 case
        if timestep_t[0] == 0:
            return torch.zeros([batch_size, 1, 1, 1], device=timestep_t.device)
        
        # Get alpha values
        t_minus_1 = torch.maximum(timestep_t - 1, torch.tensor([0], device=timestep_t.device))
        alpha_t_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, timestep_t, [batch_size, 1, 1, 1])
        alpha_t_minus_1_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, t_minus_1, [batch_size, 1, 1, 1])
        
        # DDIM variance formula: σ_t = η * sqrt((1-ᾱ_{t-1})/(1-ᾱ_t)) * sqrt(1 - ᾱ_t/ᾱ_{t-1})
        variance = eta * torch.sqrt((1 - alpha_t_minus_1_bar) / (1 - alpha_t_bar)) * torch.sqrt(1 - alpha_t_bar / alpha_t_minus_1_bar)
        
        return variance
    
    def calculate_variance_for_previous_timestep_accelerated_ddim_version(self, timestep_t, timestep_t_minus_1, eta=0.0):
        
        batch_size = timestep_t.shape[0]
        if timestep_t_minus_1[0] == 0:
            return torch.zeros([batch_size,1,1,1], device=timestep_t.device)
        
        alpha_t_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, timestep_t, [batch_size,1,1,1])
        alpha_t_minus_1_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, timestep_t_minus_1, [batch_size,1,1,1])
        
        # Calculate DDPM variance
        variance = torch.sqrt(((1 - alpha_t_minus_1_bar) / (1 - alpha_t_bar)) * (1 - (alpha_t_bar / alpha_t_minus_1_bar)))
        return eta * variance
    
    def sample_previous_timestep_DDPM_Version(self, x_t, timestep_t, pred_noise, model_variance=None):
        """
        Sample x_{t-1} from the reverse distribution given x_t and predicted noise
        
        Args:
            x_t: Current noisy image at timestep t
            t: Current timestep indices
            pred_noise: Predicted noise from model
            model_variance: Optional predicted variance from model (not used in DDPM)
            
        Returns:
            Sampled x_{t-1}
        """
        # First predict x_0 from x_t and predicted noise
        x_0_given_t = self.predict_x0_from_noise(x_t, timestep_t, pred_noise)
        
        # Compute the mean of the posterior distribution
        posterior_mean = self.calculate_mean_for_previous_timestep_DDPM_Version(x_t, timestep_t, x_0_given_t)
        
        # Compute the variance of the posterior distribution
        posterior_variance = self.calculate_variance_for_previous_timestep_DDPM_Version(timestep_t)
        posterior_log_variance = torch.log(posterior_variance)
        
        # Don't add noise at timestep 0
        if timestep_t[0] == 0:
            return posterior_mean
        
        # Sample from the posterior distribution
        noise = torch.randn_like(x_t)
        
        # Use reparameterization trick to sample
        x_t_minus_1 = posterior_mean + torch.exp(0.5 * posterior_log_variance) * noise
        
        return x_t_minus_1
    
    def sample_previous_timestep_DDIM_Version(self, x_t, timestep_t, predicted_noise, eta=1):
        """
        Fixed DDIM sampling method
        """
        # Calculate x_0_given_t first
        x_0_given_t = self.predict_x0_from_noise(x_t, timestep_t, predicted_noise)
        
        # Compute Variance First
        variance_for_previous_timestep = self.calculate_variance_for_previous_timestep_DDIM_Version(timestep_t, eta)
        
        # Use Version 2 mean calculation (it's more numerically stable)
        mean_for_previous_timestep = self.calculate_mean_for_previous_timestep_DDIM_Version2(x_t, timestep_t, predicted_noise, variance_for_previous_timestep)
        
        # Do not add noise at 0th timestep
        if timestep_t[0] == 0:
            return mean_for_previous_timestep
        
        # FIXED: Remove the square from variance
        noise = torch.randn_like(x_t)
        x_t_minus_1_DDIM_Version = mean_for_previous_timestep + variance_for_previous_timestep * noise
        
        return x_t_minus_1_DDIM_Version
    
    def sample_previous_timestep_accelerated_ddim_version(self, x_t, timestep_t, timestep_t_minus_1, predicted_noise, eta = 0.0):
        
        x_0_given_t = self.predict_x0_from_noise(x_t,timestep_t, predicted_noise)
        variance = self.calculate_variance_for_previous_timestep_accelerated_ddim_version(timestep_t, timestep_t_minus_1, eta)
        mean = self.calculate_mean_for_previous_timestep_accelerated_DDIM_Version(x_t, timestep_t, timestep_t_minus_1, x_0_given_t, variance)
        
        if timestep_t_minus_1[0] == 0:
            return mean
        
        noise = torch.randn_like(x_t)
        x_t_minus_1_accelerated = mean + variance * noise
        
        return x_t_minus_1_accelerated
    
    def sample_image_X_0_Given_T_method(self, model, shape, device, num_inference_steps=1000, return_all_steps=False):
        """
        Generate a sample using the full reverse diffusion process
        
        Args:
            model: The UNet model
            shape: Shape of the sample to generate (batch_size, channels, height, width)
            device: Device to generate on
            num_inference_steps: Number of steps to use for inference
            return_all_steps: Whether to return intermediate steps for visualization
            
        Returns:
            Generated sample(s)
        """
        model.eval()
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Store intermediate steps if requested
        intermediate_steps = [x] if return_all_steps else None
        
        # Sample using decreasing timesteps
        timesteps = torch.linspace(
            self.numberOfTimesteps - 1, 0, num_inference_steps
        ).long().to(device)
        
        # Reverse diffusion process with TQDM progress bar
        for i, t in enumerate(tqdm(timesteps, desc="Sampling", total=len(timesteps))):
            # Create batch of same timestep
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                predicted_noise = model(x, t_batch)
            
            # Sample x_{t-1} given x_t and predicted noise
            x = self.sample_previous_timestep_DDPM_Version(x, t_batch, predicted_noise)
            
            # Store this step if requested
            if return_all_steps:
                intermediate_steps.append(x)
        
        if return_all_steps:
            return intermediate_steps
        return x
    
    def sample_image_X_Using_DDIM_Sampling(self, model, shape, device, num_inference_steps=1000, eta=1):
        
        # Performing DDIM Sampling
        model.eval()
        
        # Start from Noise
        x = torch.randn(shape, device=device)
        
        # Timesteps
        timesteps = torch.linspace(self.numberOfTimesteps-1, 0, num_inference_steps).long().to(device)
        
        # Reverse Process
        for i, t in enumerate(tqdm(timesteps, desc="DDIM Sampling", total=len(timesteps))):
            
            # Creating a batch of same timestep
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                predicted_noise = model.forward(x, t_batch)
                
            # Sample x_t_minus_one
            x = self.sample_previous_timestep_DDIM_Version(x, t_batch, predicted_noise, eta)
            
        return x
    
    def sample_image_using_accelerated_ddim_sampling(self, model, shape, device, number_of_inference_steps, eta = 0.0):
        
        model.eval()
        x = torch.randn(shape, device=device)
        timesteps = self.create_timesteps_for_DDIM_accelerated_sampling(number_of_inference_steps).to(device)
        
        # Accelerated Sampling Process
        for i in tqdm(range(len(timesteps) - 1), desc='Accelerated Sampling', total=len(timesteps)-1):
            current_timestep = timesteps[i]
            next_timestep = timesteps[i+1]
            
            # Creating batch of timestep
            current_timestep_batch = torch.full((shape[0], ), current_timestep, device=device, dtype = torch.long)
            next_timestep_batch = torch.full((shape[0],), next_timestep, device=device, dtype = torch.long)
        
            # predict noise at current timestep
            with torch.no_grad():
                predicted_noise = model(x,current_timestep_batch)
                
            # De Noise
            x = self.sample_previous_timestep_accelerated_ddim_version(x, current_timestep_batch, next_timestep_batch, predicted_noise, eta)
            
        return x
            
    
    def sample_image_X_Using_Noise_Prediction_Original_DDPM_Paper(self, model, shape, device, return_all_steps=False):
        """
        DDPM 2020 sampling algorithm: iteratively denoise from pure noise using predicted noise.
        Args:
            model: The noise prediction model (e.g., UNet)
            shape: Shape of the sample to generate (batch_size, channels, height, width)
            device: Device to generate on
            return_all_steps: Whether to return all intermediate steps
        Returns:
            Final denoised sample or all steps if requested
        """
        model.eval()
        x = torch.randn(shape, device=device)
        steps = [x.clone()] if return_all_steps else None

        for t in tqdm(reversed(range(self.numberOfTimesteps)), desc="DDPM Sampling", total=self.numberOfTimesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            with torch.no_grad():
                predicted_noise = model.forward(x, t_batch)

            # Use the formula: x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1 - alpha_t) / sqrt(1 - alpha_t_bar) * predicted_noise)
            alpha_t = self.extract_value_at_timestep_t(self.alpha_t, t_batch, x.shape)
            alpha_t_bar = self.extract_value_at_timestep_t(self.alpha_t_bar, t_batch, x.shape)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t_bar = torch.sqrt(1 - alpha_t_bar)
            coef = (1 - alpha_t) / sqrt_one_minus_alpha_t_bar
            mean = (1 / sqrt_alpha_t) * (x - coef * predicted_noise)
            var = self.calculate_variance_for_previous_timestep_DDPM_Version(t_batch)
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(var) * noise
            else:
                x = mean
            if return_all_steps:
                steps.append(x.clone())
        if return_all_steps:
            return steps
        return x
    
    # Additional 2 methods
    def sample_image_using_accelerated_ddim_sampling_with_beta_schedule(self, model, shape, device, number_of_inference_steps, eta=0.0, spacing="linear"):
        """
        DDIM sampling that properly separates beta schedule from timestep spacing
        """
        model.eval()
        x = torch.randn(shape, device=device)
        timesteps = self.create_timesteps_for_DDIM_accelerated_sampling(number_of_inference_steps, spacing).to(device)
        
        print(f"Using beta schedule: {self.betas[:10]}...")  # Debug print
        print(f"Using timestep spacing: {spacing}")
        print(f"Selected timesteps: {timesteps[:10]}...")
        
        for i in tqdm(range(len(timesteps) - 1), desc=f'DDIM Sampling (β-schedule, {spacing} spacing)', total=len(timesteps)-1):
            current_timestep = timesteps[i]
            next_timestep = timesteps[i+1]
            
            current_timestep_batch = torch.full((shape[0],), current_timestep, device=device, dtype=torch.long)
            next_timestep_batch = torch.full((shape[0],), next_timestep, device=device, dtype=torch.long)
        
            with torch.no_grad():
                predicted_noise = model(x, current_timestep_batch)
                    
            x = self.sample_previous_timestep_accelerated_ddim_version(x, current_timestep_batch, next_timestep_batch, predicted_noise, eta)
                
        return x