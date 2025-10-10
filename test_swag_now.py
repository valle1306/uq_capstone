
import torch
import sys
import os
sys.path.insert(0, '/scratch/hpl14/uq_capstone/src')
from model_utils import UNet
from swag import SWAG
import numpy as np

print('Testing FIXED SWAG...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Load one test sample
npz_dir = '/scratch/hpl14/uq_capstone/data/brats/test'
test_files = sorted([f for f in os.listdir(npz_dir) if f.endswith('.npz')])
data = np.load(os.path.join(npz_dir, test_files[0]))
image = torch.from_numpy(data['image']).unsqueeze(0).to(device)
mask = torch.from_numpy(data['mask']).unsqueeze(0).to(device)

print(f'Loaded: {test_files[0]}, shape: {image.shape}\n')

# Test different max_var values
for max_var in [1.0, 5.0, 10.0]:
    print(f'max_var={max_var}:', end=' ')
    
    base_model = UNet(in_channels=1, num_classes=1, dropout_rate=0.0)
    swag_model = SWAG(base_model, max_num_models=20, max_var=max_var)
    
    checkpoint = torch.load('/scratch/hpl14/uq_capstone/runs/swag/swag_model.pth', 
                           map_location=device, weights_only=False)
    swag_model.n_models = checkpoint['n_models']
    swag_model.mean = checkpoint['mean'].to(device)
    swag_model.sq_mean = checkpoint['sq_mean'].to(device)
    swag_model.cov_mat_sqrt = [d.to(device) for d in checkpoint['cov_mat_sqrt']]
    swag_model.max_num_models = checkpoint['max_num_models']
    swag_model.to(device)
    
    try:
        mean_pred, uncertainty = swag_model.predict_with_uncertainty(
            image, n_samples=5, scale=0.5
        )
        
        pred_binary = (mean_pred > 0.5).float()
        intersection = (pred_binary * mask).sum()
        dice = (2. * intersection) / (pred_binary.sum() + mask.sum() + 1e-8)
        
        print(f'Dice={dice.item():.4f}, Unc={uncertainty.mean():.6f}, NaN={torch.isnan(uncertainty).any().item()}')
    except Exception as e:
        print(f'FAILED: {str(e)[:80]}')

print('\nTest complete!')
