import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy

# CNN processes visual input (RGB images) to extract spatial features like object position
# MLP processes state input (e.g., joint angles, positions), and both outputs are combined and used by SAC for decision-making.


# This class extracts features from both image (CNN) and state vector (MLP)
# It combines the outputs of both to provide a joint representation for the agent
class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim=features_dim)
        
        # Get actual image dimensions from observation space
        image_space = observation_space['image']
        image_shape = image_space.shape
        print(f"Image shape from observation space: {image_shape}")
        
        # CNN for extracting visual features from 64x64 RGB images
        # Used smaller kernel sizes appropriate for 64x64 images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),  
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()    # Output is flattened to a vector
        )
        
        
        # Computing output dimension of CNN using a dummy input
        with torch.no_grad():
            # Create a properly shaped sample image - channels first for PyTorch CNN
            sample_img = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
            print(f"Sample image shape: {sample_img.shape}")
            cnn_output = self.cnn(sample_img)
            cnn_out_dim = cnn_output.shape[1]
            print(f"CNN output dimension: {cnn_out_dim}")
            
        # MLP for processing the vector-based state (joint angles, positions, etc.)
        self.mlp = nn.Sequential(
            nn.Linear(observation_space['state'].shape[0], 128),
            nn.ReLU()
        )
        
        # Final feature dimension is combination of CNN and MLP outputs
        self._features_dim = cnn_out_dim + 128
        print(f"Total features dimension: {self._features_dim}")
        
    def forward(self, observations):
        # Normalizing the image and pass it through the CNN
        image = observations['image'].float() / 255.0
        # Concatenating CNN and MLP outputs to form a combined feature vector
        return torch.cat([
            self.cnn(image),
            self.mlp(observations["state"])
        ], dim=1)
        
# This policy tells SAC to use the custom CNN+MLP feature extractor defined above
class CustomSACPolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs={},
            **kwargs
        )