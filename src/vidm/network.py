import torch
import torch.nn as nn
import torch.nn.functional as F

from vidm.cunet import UNetModel
from vidm.encoder import EncoderUNetModel

from mmedit.models.backbones.sr_backbones.basicvsr_net import SPyNet

class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class ComplexUModel(nn.Module):
    """
    A UNetModel that performs constant input

    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        channel_mult=(1, 2, 4, 8),
        num_heads=1,
        num_head_channels=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        diffusion_timesteps=None,
        video_timesteps=None,
        spynet_pretrained=None,
    ):
        super().__init__()
        self.encoder = EncoderUNetModel(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            resblock_updown=resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            diffusion_timesteps=diffusion_timesteps,
            video_timesteps=video_timesteps,
        )

        self.unet = UNetModel(
            image_size=image_size,
            in_channels=in_channels + 2,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            resblock_updown=resblock_updown,
            use_scale_shift_norm=use_scale_shift_norm,
            diffusion_timesteps=diffusion_timesteps,
            video_timesteps=video_timesteps,
        )
        
        self.spynet = SPyNet(pretrained=spynet_pretrained)

    def forward(self, x, dt, vt, x_mins_1, x_0):
        flow = self.spynet(x_0, x_mins_1)
        h = torch.cat([x, flow], dim=1)
        
        residual = x_0
        hs = []
        for module, cmodule in zip(self.encoder.input_blocks, self.unet.input_blocks):
            residual = module(residual, dt, vt)
            h = residual + cmodule(h, dt, vt) * 0.2
            hs.append(h)
        h = self.unet.middle_block(h, dt, vt)
        for module in self.unet.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, dt, vt)
        return self.unet.out(h)
