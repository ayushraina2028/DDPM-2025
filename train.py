# train.py
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import sys
import argparse

sys.path.append("/home/jovyan/beomi/ayushraina/ddpm_pipeline")
from diffusion.scheduler import Diffusion_Scheduler
from diffusion.trainer import trainDiffusionModel_DDPM
from models.unet import SimpleUNetModel

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train diffusion model")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./diffusion_output", help="Output directory")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--sampling_method", type=str, default="ddpm_predicted_x0", 
                        choices=["ddpm_predicted_x0", "ddpm_original", "ddim_sampling"],
                        help="Sampling method for visualizations")
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(
        root='data/afhq',
        transform=transform
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    # Initialize model, scheduler, optimizer
    model = SimpleUNetModel(
        inChannels=3,
        outChannels=3,
        baseChannels=64,
        timeEmbeddingDimension=128,
        depth=4
    )
    
    scheduler = Diffusion_Scheduler(beta_schedule='quadratic')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    trained_model = trainDiffusionModel_DDPM(
        Model=model,
        Scheduler=scheduler,
        TrainDataLoader=train_dataloader,
        Optimizer=optimizer,
        Device=device,
        OutputDirectory=args.output_dir,
        NumberOfEpochs=args.epochs,
        SaveIntervals=1,
        LogIntervals=100,
        ImageSize=args.image_size,
        SamplingMethod=args.sampling_method
    )
    
    # Save final model
    torch.save(trained_model.state_dict(), os.path.join(args.output_dir, "final_model.pt"))
    print(f"Model saved to {os.path.join(args.output_dir, 'final_model.pt')}")

if __name__ == "__main__":
    main()