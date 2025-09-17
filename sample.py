# sample.py
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import os
import sys
from tqdm import tqdm  # ✅ ADDED

sys.path.append("/home/jovyan/beomi/ayushraina/ddpm_pipeline")
# from diffusion.scheduler import Diffusion_Scheduler
from diffusion.schedule_self import Diffusion_Scheduler
from models.unet import SimpleUNetModel

def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained diffusion model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--output_dir", type=str, default="./Generations", help="Directory to save samples")
    parser.add_argument("--inference_steps", type=int, default=1000, help="Number of inference steps")
    parser.add_argument("--save_grid", action="store_true", help="Save a grid of all samples")
    parser.add_argument("--sampling_method", type=str, default="ddpm_predicted_x0", 
                        choices=["ddpm_predicted_x0", "ddpm_original", "ddim_sampling", "accelerated_ddim_sampling"], 
                        help="Sampling method to use (ddpm_predicted_x0 or ddpm_original)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = SimpleUNetModel(
        inChannels=3,
        outChannels=3,
        baseChannels=64,
        timeEmbeddingDimension=128,
        depth=3
    )

    # ✅ FIXED: Secure loading (weights_only=True)
    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    scheduler = Diffusion_Scheduler()
    
    # Set up shape for batch generation
    sample_shape = (args.num_samples, 3, args.image_size, args.image_size)
    
    print(f"Generating {args.num_samples} samples with {args.inference_steps} inference steps...")
    print(f"Using sampling method: {args.sampling_method}")
    
    # Generate samples based on chosen sampling method
    if args.sampling_method == "ddpm_predicted_x0":
        samples = scheduler.sample_image_X_0_Given_T_method(
            model=model,
            shape=sample_shape,
            device=device,
            num_inference_steps=args.inference_steps,
            return_all_steps=False
        )
    elif args.sampling_method == "ddpm_original":
        # For the original DDPM sampling method, we use the full number of timesteps
        # instead of the inference_steps parameter
        samples = scheduler.sample_image_X_Using_Noise_Prediction_Original_DDPM_Paper(
            model=model,
            shape=sample_shape,
            device=device,
            return_all_steps=False
        )
    elif args.sampling_method == "ddim_sampling":
        # Perform DDIM Sampling in this case
        eta = 0
        samples = scheduler.sample_image_X_Using_DDIM_Sampling(model=model, shape=sample_shape, device=device, num_inference_steps=args.inference_steps, eta=eta)
    elif args.sampling_method == "accelerated_ddim_sampling":
        eta = 0.4
        samples = scheduler.sample_image_using_accelerated_ddim_sampling_improved(model=model, shape=sample_shape, device=device, number_of_inference_steps=args.inference_steps, eta=eta)        

        
    # Denormalize samples from [-1, 1] to [0, 1] if needed
    # (Assuming the model outputs in [-1, 1] range)
    samples = (samples.clamp(-1, 1) + 1) / 2
    
    # Optionally save a grid of all samples
    if args.save_grid:
        print("Creating image grid...")
        grid = torchvision.utils.make_grid(samples, nrow=int(args.num_samples**0.5))
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        grid_filename = f"samples_grid_{args.sampling_method}.png"
        plt.savefig(os.path.join(args.output_dir, grid_filename), bbox_inches='tight')
        plt.close()
    
    print(f"Successfully generated {args.num_samples} samples using {args.sampling_method} method in {args.output_dir}")

if __name__ == "__main__":
    main()