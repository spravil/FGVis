import argparse
from collections.abc import Iterable
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms
from tqdm import tqdm


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def replace_module(model, key, new_module):
    keys = key.split(".")
    last_key = keys[-1]
    keys = keys[:-1]

    current_module = model
    for token in keys:
        if token.isdigit():
            token = int(token)
            if token < len(current_module):
                current_module = current_module[token]
            else:
                raise Exception(f"Index out of range: {token}")
        elif isinstance(current_module, Iterable) and token in current_module:
            current_module = current_module[token]
        elif hasattr(current_module, token):
            current_module = getattr(current_module, token)
        else:
            raise Exception()

    if last_key.isdigit():
        last_key = int(last_key)
        current_module[last_key] = new_module
    elif isinstance(current_module, Iterable) and last_key in current_module:
        current_module[last_key] = new_module
    elif hasattr(current_module, last_key):
        setattr(current_module, last_key, new_module)
    else:
        raise Exception()


def load_model(model_name, device):
    if model_name.startswith("vgg"):
        model = getattr(models, model_name)(pretrained=True)
    else:
        model = timm.create_model(model_name, pretrained=True)

    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    return model


class DefenseFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bu, bl):
        ctx.save_for_backward(x, bu, bl)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, bu, bl = ctx.saved_tensors
        grad_x = grad_output * torch.logical_and(x <= bu, x >= bl)
        return grad_x, None, None


class DefenseLayer(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        bu = torch.clone(output)
        bu[bu < 0] = 0
        self.register_buffer("bu", bu)
        bl = torch.clone(output)
        bl[bl > 0] = 0
        self.register_buffer("bl", bl)

    def forward(self, x):
        return DefenseFunction.apply(x, self.bu, self.bl)


def run(args):
    device = get_device()

    model = load_model(args.model, device)
    non_linearity_outputs = {}

    def register_relu_callback(module, input, output, name):
        non_linearity_outputs[name] = output.detach()

    modules = model.named_modules()
    modules_to_replace = []
    act_layer = torch.nn.ReLU
    for name, module in modules:
        if isinstance(module, act_layer):
            modules_to_replace.append((name, module))
            module.register_forward_hook(partial(register_relu_callback, name=name))

    img = Image.open(args.image_path)
    transform = timm.data.create_transform(input_size=224, interpolation="bicubic", crop_pct=1.0)
    original_img = transform(img)
    img = original_img.unsqueeze(0).to(device)
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255],
    )
    original_img = inv_normalize(original_img).permute(1, 2, 0)

    zero_img = transform(Image.new("RGB", (224, 224), (0, 0, 0))).unsqueeze(0).to(device)
    mask = torch.ones((1, 3, 224, 224), dtype=torch.float32).to(device)
    mask = Variable(mask, requires_grad=True)

    optimizer = torch.optim.SGD([mask], lr=args.learning_rate)

    out = model(img)
    target = torch.nn.Softmax(dim=1)(out)
    category = np.argmax(target.cpu().data.numpy())
    print(f"Category with highest probability: {category}")

    print(f"Add layer clipping for {len(non_linearity_outputs)} non linearities")
    for name, module in modules_to_replace:
        out = non_linearity_outputs[name]
        defense = DefenseLayer(out)
        defense.requires_grad = True
        replace_module(model, name, torch.nn.Sequential(module, defense))

    for i in tqdm(range(args.max_iterations)):
        upsampled_mask = mask
        upsampled_mask = upsampled_mask.expand(
            1, upsampled_mask.size(1), upsampled_mask.size(2), upsampled_mask.size(3)
        )
        perturbated_input = img.mul(upsampled_mask) + zero_img.mul(1 - upsampled_mask)
        perturbated_input += torch.randn_like(perturbated_input) * 0.1

        outputs = torch.nn.functional.softmax(model(perturbated_input), dim=1)
        loss = args.mask_weight * torch.norm(mask, 1) + outputs[0, category]

        optimizer.zero_grad()
        loss.backward()
        max_grad = mask.grad.abs().max()
        mask.grad = mask.grad / max_grad
        optimizer.step()
        mask.data.clamp_(0, 1)

        new_category = np.argmax(outputs[0].cpu().data.numpy())
        if new_category != category:
            print(f"Stop early because prediction switched from {category} to {new_category}")
            break

    print("Done.")
    return mask, original_img


def main():
    parser = argparse.ArgumentParser(description="Run FGVis.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model", type=str, default="vgg16", help="Model name to use (Default: vgg16).")
    parser.add_argument("--max_iterations", type=int, default=500, help="Maximum number of iterations (Default: 500).")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate (Default: 0.1).",
    )
    parser.add_argument(
        "--mask_weight",
        type=float,
        default=-1e-9,
        help="Mask weight (Default: -1e-9).",
    )
    args = parser.parse_args()

    mask, original_img = run(args)
    mask = mask.cpu().detach()
    reshaped_mask = mask[0].permute(1, 2, 0)
    reshaped_mask[reshaped_mask > 0.99] = 1

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].axis("off")
    axs[0].imshow(original_img)

    a = original_img.mul(1 - reshaped_mask) + (torch.ones_like(original_img) * 0.5).mul(reshaped_mask)
    a -= a.min()
    a /= a.max()
    axs[1].imshow(a)
    axs[1].axis("off")

    axs[2].imshow(1 - reshaped_mask.mean(-1))
    axs[2].axis("off")

    plt.tight_layout()

    plt.savefig("output.png")
    print("Save result to output.png")


if __name__ == "__main__":
    main()
