import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ==================== U-Net for Segmentation ====================

class DoubleConv(nn.Module):
    """Double convolution block for U-Net"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling block with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block with transpose conv then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 to match x2 size if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for image segmentation
    with optional dropout for MC Dropout UQ
    """
    def __init__(self, in_channels=1, num_classes=1, dropout_rate=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        
        # Dropout layers
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)
        else:
            self.dropout = None
        
        # Decoder
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        
        # Output
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply dropout if enabled
        if self.dropout is not None:
            x5 = self.dropout(x5)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


# ==================== Loss Functions ====================

class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        dice = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        return 1 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice + Binary Cross Entropy Loss"""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, logits, targets):
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


# ==================== Classification Models ====================

def get_resnet50(num_classes=2, pretrained=True):
    import torchvision.models as models
    model = models.resnet50(pretrained=pretrained)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for x,y in tqdm(dataloader, leave=False):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def predict_probs(model, dataloader, device):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for x,y in tqdm(dataloader, leave=False):
            x = x.to(device)
            logits = model(x)
            p = F.softmax(logits, dim=1)[:,1].cpu().numpy()
            probs.extend(p.tolist())
            labels.extend(y.numpy().tolist())
    return probs, labels

def nll_brier_ece(probs, labels, n_bins=20):
    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss
    probs = np.array(probs)
    labels = np.array(labels)
    nll = log_loss(labels, probs)
    brier = brier_score_loss(labels, probs)
    # ECE
    bins = np.linspace(0.0, 1.0, n_bins+1)
    bin_ids = np.digitize(probs, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(conf - acc)
    return nll, brier, ece

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

def fit_temperature(model, valid_loader, device):
    # Collect logits and labels
    model.eval()
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for x,y in valid_loader:
            x = x.to(device)
            logits = model(x)
            logits_list.append(logits.cpu())
            labels_list.append(y)
    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    temp_model = TemperatureScaling()
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)

    nll_criterion = nn.CrossEntropyLoss()

    def eval_fn():
        optimizer.zero_grad()
        scaled = temp_model(logits)
        loss = nll_criterion(scaled, labels)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    return temp_model.temperature.item()
