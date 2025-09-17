import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import os 
import sys
import time 
from tqdm import tqdm 

sys.path.append("/home/jovyan/beomi/ayushraina/ddpm_pipeline")
from diffusion.scheduler import Diffusion_Scheduler
from models.unet import SimpleUNetModel

def trainDiffusionModel_DDPM(Model: SimpleUNetModel, Scheduler: Diffusion_Scheduler, TrainDataLoader, Optimizer, Device, OutputDirectory = "./TrainingProgress", NumberOfEpochs=50, SaveIntervals=10, LogIntervals=100, SampleSize=10, ImageSize=64, SamplingMethod='ddpm_predicted_x0'):
    
    # Training Loop for DDPM
    """
    Args:
        model: UNet model for noise prediction
        scheduler: Diffusion scheduler
        train_dataloader: DataLoader containing training images
        optimizer: Optimizer for updating model parameters
        device: Device to train on (CPU or GPU)
        output_dir: Directory to save visualizations and checkpoints
        num_epochs: Number of epochs to train for
        save_interval: Intervals (in epochs) to save visualizations
        log_interval: How often to log progress (in batches)
        sample_size: Number of images to generate for visualization
    """
    
    # Create the output directory for checking training process if it does not exists already
    os.makedirs(OutputDirectory, exist_ok=True)
    
    # Move model to device
    Model.to(Device)
    
    # Get fixed noise for consistent visualization
    FixedNoise = torch.randn(SampleSize, 3, ImageSize, ImageSize, device=Device)
    
    # Get a batch of real images
    # batch = next(iter(TrainDataLoader))
    # if isinstance(batch, (list, tuple)):
    #     RealBatchOfImages = batch[0][:SampleSize].to(Device)
    # else:
    #     RealBatchOfImages = batch[:SampleSize].to(Device)
    
    # Tracking
    Losses = []
    EpochLosses = []
    
    # Training Loop
    for epoch in range(NumberOfEpochs):
        
        # Get a batch of real images
        batch = next(iter(TrainDataLoader))
        if isinstance(batch, (list, tuple)):
            RealBatchOfImages = batch[0][:SampleSize].to(Device)
        else:
            RealBatchOfImages = batch[:SampleSize].to(Device)
        
        Model.train()
        EpochStartTime = time.time()
        RunningLoss = 0.0
        BatchCount = 0
        
        # Progress Bar for current epoch
        import sys
        ProgressBar = tqdm(
            TrainDataLoader,
            desc=f"Epoch {epoch+1}/{NumberOfEpochs}",
            disable=not sys.stdout.isatty(),   # disables tqdm if output is not a terminal
            dynamic_ncols=True
        )
                
        for BatchIndex, CurrentBatch in enumerate(ProgressBar):
            
            # Handle both tensor and tuple cases (image, label)
            if isinstance(CurrentBatch, (list, tuple)):
                X_0 = CurrentBatch[0].to(Device)  # Extract just the images
            else:
                X_0 = CurrentBatch.to(Device)
            
            BatchSize = X_0.shape[0]
            
            # Sample timesteps from uniform dist
            timesteps = torch.randint(0, Scheduler.numberOfTimesteps, (BatchSize,), device=Device)
            
            # Forward Process
            X_t, TrueNoise = Scheduler.addNoiseKernel(X_0, timesteps)
            
            # Predict the noise using Unet Model
            Predicted_Noise = Model.forward(X_t, timesteps)
            
            # Calculate Loss (MSE)
            Loss = F.mse_loss(Predicted_Noise, TrueNoise)
            
            # Update Parameter
            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()

            # Tracking the Losses
            RunningLoss += Loss.item()
            BatchCount += 1
            
            # Update Progress Bar
            ProgressBar.set_postfix({"loss: ": Loss.item()})
            
            # Log Progress
            if(BatchIndex + 1) % LogIntervals == 0:
                
                AverageLoss = RunningLoss / BatchCount
                Losses.append(AverageLoss)
                # print(f"Epoch {epoch+1}, Batch {BatchIndex+1}: Loss = {AverageLoss:.6f}")
                RunningLoss = 0
                BatchCount = 0
                
        EpcohTime = time.time() - EpochStartTime
        EpochAverageLoss = RunningLoss / max(BatchCount,1)
        EpochLosses.append(EpochAverageLoss)
        print(f"Epoch {epoch+1} completed in {EpcohTime:.2f}s, Avg Loss: {EpochAverageLoss:.6f}")
            
        # Save model checkpoint
        checkpoint_path = os.path.join(OutputDirectory, f"model_parameters.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': Model.state_dict(),
            'optimizer_state_dict': Optimizer.state_dict(),
            'loss': EpochAverageLoss,
        }, checkpoint_path)

        # Generate visualizations at specified intervals
        if (epoch + 1) % SaveIntervals == 0 or epoch == 0 or epoch == NumberOfEpochs - 1:
            visualization_path = os.path.join(OutputDirectory, f"progress.png")
            generate_and_save_visualizations(
                Model, 
                Scheduler, 
                RealBatchOfImages,
                Device, 
                visualization_path,
                epoch + 1,
                ImageSize,
                SamplingMethod
            )
            
            # Also save loss plot
            loss_plot_path = os.path.join(OutputDirectory, f"loss_plot.png")
            save_loss_plots(
                Losses, 
                EpochLosses, 
                loss_plot_path, 
                epoch + 1,
                LogIntervals,
                len(TrainDataLoader)
            )
        
    print("Training complete!")
    return Model

def generate_and_save_visualizations(Model, Scheduler, RealImages, Device, SavePath, CurrentEpoch, ImageSize=64, sampling_method="ddpm_predicted_x0"):
    """
    Generate visualizations of model performance during training and save to disk.
    
    Args:
        Model: The UNet model
        Scheduler: The diffusion scheduler
        RealImages: Batch of real images for comparison
        Device: Device to run generation on
        SavePath: Path to save visualization
        CurrentEpoch: Current epoch number
        ImageSize: Size of images
        sampling_method: Which sampling method to use (default: ddpm_predicted_x0)
    """
    Model.eval()
    sample_size = min(10, RealImages.shape[0])
    
    # Create a figure with rows: 
    # 1. Original images
    # 2. Reconstructed images from moderate noise (t=500)
    # 3. Generated images (from pure noise)
    fig, axes = plt.subplots(3, sample_size, figsize=(sample_size*3, 9))
    
    with torch.no_grad():
        # Row 1: Display real images
        for i in range(sample_size):
            # Denormalize from [-1, 1] to [0, 1]
            img = RealImages[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
            axes[0, i].imshow(torch.clamp(img, 0, 1).numpy())
            axes[0, i].set_title("Real Image")
            axes[0, i].axis('off')
        
        # Row 2: Reconstruct from moderate noise (random t per image)
        t_reconstruct = torch.randint(100, 1000, (sample_size,), device=Device)
        
        # Add noise to the real images
        x_t, true_noise = Scheduler.addNoiseKernel(RealImages[:sample_size], t_reconstruct)
        
        # Predict the noise
        predicted_noise = Model(x_t, t_reconstruct)
        
        # Estimate x_0 from the predicted noise
        predicted_x_0 = Scheduler.predict_x0_from_noise(x_t, t_reconstruct, predicted_noise)
        
        # Display reconstructions
        for i in range(sample_size):
            img = predicted_x_0[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
            axes[1, i].imshow(torch.clamp(img, 0, 1).numpy())
            axes[1, i].set_title(f"Reconstruction at t={t_reconstruct[i].item()}")
            axes[1, i].axis('off')
        
        # Row 3: Generate new images from pure noise using the sampling process
        # Use the proper sampling method based on parameter
        if sampling_method == "ddpm_predicted_x0":
            generated_samples = Scheduler.sample_image_X_0_Given_T_method(
                model=Model,
                shape=(sample_size, 3, ImageSize, ImageSize),
                device=Device,
                num_inference_steps=1000,  # Use fewer steps for visualization during training
                return_all_steps=False
            )
        else:  # Use original DDPM method
            generated_samples = Scheduler.sample_image_X_Using_Noise_Prediction_Original_DDPM_Paper(
                model=Model,
                shape=(sample_size, 3, ImageSize, ImageSize),
                device=Device,
                return_all_steps=False
            )
                
        # Display the generated images
        for i in range(sample_size):
            img = generated_samples[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
            axes[2, i].imshow(torch.clamp(img, 0, 1).numpy())
            axes[2, i].set_title(f"Generated ({sampling_method})")
            axes[2, i].axis('off')
    
    plt.suptitle(f"Training Progress - Epoch {CurrentEpoch}")
    plt.tight_layout()
    plt.savefig(SavePath)
    plt.close()
    
    print(f"Visualization saved to {SavePath}")
    
def save_loss_plots(Losses, EpochLosses, SavePath, CurrentEpoch, LogIntervals, BatchesPerEpoch):
    """
    Create and save plots showing training loss progress.
    """
    plt.figure(figsize=(10, 5))
    
    # Plot batch losses
    batch_indices = [i * LogIntervals for i in range(len(Losses))]
    plt.plot(batch_indices, Losses, label='Batch Loss', alpha=0.6)
    
    # Plot epoch losses
    epoch_indices = [i * BatchesPerEpoch for i in range(len(EpochLosses))]
    plt.plot(epoch_indices, EpochLosses, 'ro-', label='Epoch Loss')
    
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title(f'Training Loss - Up to Epoch {CurrentEpoch}')
    plt.legend()
    plt.savefig(SavePath)
    plt.close()