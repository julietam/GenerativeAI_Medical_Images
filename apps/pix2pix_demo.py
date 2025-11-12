import argparse
import gradio as gr
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np

from src.models.pix2pix import UNetGenerator

try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
    HAS_TM = True
except Exception:
    HAS_TM = False
    from src.utils.metrics import ssim as ssim_fn, psnr as psnr_fn


def load_generator(checkpoint: str, in_c: int = 1, out_c: int = 1, device: str = "cpu") -> torch.nn.Module:
    G = UNetGenerator(in_channels=in_c, out_channels=out_c)
    state = torch.load(checkpoint, map_location=device)
    # support either raw state_dict or dict with 'model' key
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    G.load_state_dict(state)
    G.eval().to(device)
    return G


def preprocess_pil_to_tensor(img: Image.Image, img_size: int = 256) -> torch.Tensor:
    # convert to grayscale float tensor in [-1,1]
    img = img.convert("L").resize((img_size, img_size))
    x = np.array(img).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = torch.from_numpy(x)[None, None, :, :]
    return x


def postprocess_tensor_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    # from [-1,1] to [0,1]
    x = (x + 1) / 2
    x = x.clamp(0, 1)
    arr = (x[0].numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def make_interface(G: torch.nn.Module, device: str = "cpu", img_size: int = 256):
    title = "Pix2Pix MRI T1→T2 Demo (Live Metrics)"
    description = (
        "Sube una imagen T1 (PNG/JPG). Opcionalmente sube la T2 real (ground truth).\n"
        "El modelo genera una T2 sintética y, si proporcionas T2 real, se calculan SSIM/PSNR al vuelo."
    )

    if HAS_TM:
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    def infer(image: Image.Image, gt_image: Image.Image | None):
        with torch.no_grad():
            x = preprocess_pil_to_tensor(image, img_size=img_size).to(device)
            y = G(x)
            out_img = postprocess_tensor_to_pil(y)
            ssim_val = None
            psnr_val = None
            if gt_image is not None:
                gt = preprocess_pil_to_tensor(gt_image, img_size=img_size).to(device)
                y01 = (y + 1) / 2
                gt01 = (gt + 1) / 2
                if HAS_TM:
                    ssim_val = float(ssim_metric(y01, gt01).item())
                    psnr_val = float(psnr_metric(y01, gt01).item())
                else:
                    from src.utils.metrics import ssim as ssim_fn, psnr as psnr_fn
                    ssim_val = float(ssim_fn(y01, gt01).item())
                    psnr_val = float(psnr_fn(y01, gt01).item())
            return out_img, ssim_val, psnr_val

    with gr.Blocks() as demo:
        gr.Markdown(f"# {title}\n{description}")
        with gr.Row():
            in_img = gr.Image(type="pil", label="Entrada T1 (grayscale)")
            gt_img = gr.Image(type="pil", label="T2 Ground Truth (opcional)")
        with gr.Row():
            out_img = gr.Image(type="pil", label="Salida T2 (generada)")
        with gr.Row():
            ssim_box = gr.Number(label="SSIM vs GT", value=None)
            psnr_box = gr.Number(label="PSNR vs GT", value=None)
        btn = gr.Button("Generar")
        btn.click(fn=infer, inputs=[in_img, gt_img], outputs=[out_img, ssim_box, psnr_box])
    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio Pix2Pix MRI demo")
    parser.add_argument("--checkpoint", type=str, required=True, help="Ruta al checkpoint de G (Pix2Pix)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    G = load_generator(args.checkpoint, device=args.device)
    demo = make_interface(G, device=args.device, img_size=args.img_size)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
