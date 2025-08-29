import torch
from torch import nn
from .module import PoseTransformer
from fusion_net.module import Mlp
from fusion_net.module import DenseFC
from fusion_net.module import ResidualFC
from fusion_net.module import DenseGCN, adj_mx_from_skeleton
from common.arguments import parse_args

args = parse_args()

class FusionNet(nn.Module):
    def __init__(self, num_frame, num_joints, hidden_channel, out_channel, regression_head="dense"):
        super(FusionNet, self).__init__()

        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.adj_mx_from_skeleton = adj_mx_from_skeleton()

        # Feature extractor
        self.feature_extractor = PoseTransformer(num_frame=num_frame, num_joints=num_joints,
            in_chans=3, out_chans=hidden_channel, embed_dim_ratio=32,
            depth=4, num_heads=8, mlp_ratio=2.0, qkv_bias=True, qk_scale=None,
            drop_path_rate=0.1)  # default (good) depth=4, num_heads=8
        
        # Calculate input features for regression head
        feature_dim = hidden_channel * args.num_proposals

        # Regression head
        if regression_head == "mlp":
            self.regression_head = Mlp(
                in_features=feature_dim,
                out_features=out_channel)

        elif regression_head == "dense":
            self.regression_head = DenseFC(
                in_channel=feature_dim,
                out_channel=out_channel,
                linear_size=1024,  # 2048,
                num_stage=2,  # 2
                p_dropout=0.01)

        elif regression_head == "residual":
            self.regression_head = ResidualFC(
                in_channel=feature_dim,
                out_channel=out_channel,
                linear_size=512,  # 2048,
                num_stage=2,  # 2
                p_dropout=0.01)
            
        elif regression_head == "dense_gcn":
            self.regression_head = DenseGCN(
                adj=self.adj_mx_from_skeleton,
                in_dim=hidden_channel * args.num_proposals,
                out_dim=out_channel,
                inter_dim=32,
                num_layer=2)

        else:
            raise ValueError(f"Unsupported regression head: {self.regression_head}.")

    def forward(self, x):
        # feature extractor (PoseTransformer)
        B, H, F, J, _ = x.shape # B: batch_size, H: Hypothesis, F: Frames, J: Joints
        features_extracted = []
        for proposal in range(args.num_proposals):
            output_extractor = self.feature_extractor(x[:, proposal])
            features_extracted.append(output_extractor)
        features_extracted = torch.cat(features_extracted, dim=3) # → (B, F, J, H*C)
        print("features_extracted.shape:", features_extracted.shape)
        # Reshape to (batch_size, num_joints, feature_dim)
        features_extracted = features_extracted[:, 0].reshape(B, J, -1)
        # Regression Head
        output = self.regression_head(features_extracted)
        # Reshape output to (batch_size, frames, joints, channels)
        output = output.reshape(B, J, -1)  # First ensure it's (batch_size, joints, channels)
        output = output.reshape(B, 1, J, -1)  # Add frame dimension
        output = output.expand(B, F, J, -1)  # Expand to all frames
        return output
        
if __name__ == '__main__':
    frame = 243
    c = 2
    num_joint = 17

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    encoder = FusionNet(num_frame=243, num_joints=17, hidden_channel=24,
                        out_channel=3, regression_head="dense_gcn").cuda()
    print(encoder)
    # norm_1 = nn.LayerNorm(frame*5)

    input = torch.randn(10, 5, 243, 17, 3, dtype=torch.float32).cuda()
    out_put = encoder(input)
    print(out_put.shape)
    print('Done!')