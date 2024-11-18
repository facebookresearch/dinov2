"""DINOV2 model converter to onnx."""
import torch
import argparse
import os
import sys
from pathlib import Path
current_path = Path(__file__).resolve()
parent_path = current_path.parent.parent.as_posix()
sys.path.insert(0, parent_path)
import hubconf


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tensor):
        ff = self.model(tensor)
        return ff

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="dinov2_vits14", help="dinov2 model name")
parser.add_argument(
    "--image_height", type=int, default=518, help="input image height, must be a multiple of patch_size"
)
parser.add_argument(
    "--image_width", type=int, default=518, help="input image height, must be a multiple of patch_size"
)
parser.add_argument(
    "--patch_size", type=int, default=14, help="dinov2 model patch size, default is 16"
)
args = parser.parse_args()


if __name__ == "__main__":

    assert args.image_height % args.patch_size == 0, f"image height must be multiple of {args.patch_size}, but got {args.image_height}"
    assert args.image_width % args.patch_size == 0, f"image width must be multiple of {args.patch_size}, but got {args.image_height}"

    model = Wrapper(hubconf.dinov2_vits14(for_onnx=True)).to("cpu")
    model.eval()

    dummy_input = torch.rand([1, 3, args.image_height, args.image_width]).to("cpu")
    dummy_output = model(dummy_input)

    torch.onnx.export(
        model,
        dummy_input,
        args.model_name + ".onnx",
        input_names = ["input"]
    )