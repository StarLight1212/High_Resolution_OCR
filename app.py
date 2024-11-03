import gradio as gr
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from omegaconf import OmegaConf
from sampler import Sampler
from utils import util_image
from basicsr.utils.download_util import load_file_from_url


# Initialize models
def init_models():
    # Initialize OCR
    tokenizer = AutoTokenizer.from_pretrained('./ocr_plugin/', trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    ocr_model = AutoModel.from_pretrained(
        './ocr_plugin/',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map='cuda',
        use_safetensors=True,
        pad_token_id=tokenizer.eos_token_id
    ).eval().cuda()

    return tokenizer, ocr_model


# SinSR configurations
def get_configs(model, colab):
    config_paths = {
        'SinSR': './configs/SinSR.yaml',
        'ResShift': './configs/realsr_swinunet_realesrgan256.yaml'
    }
    if colab:
        config_paths.update({
            'SinSR': '/content/SinSR/configs/SinSR.yaml',
            'ResShift': '/content/SinSR/configs/realsr_swinunet_realesrgan256.yaml'
        })

    configs = OmegaConf.load(config_paths[model])
    task = "realsrx4" if model == 'ResShift' else None

    # Setup checkpoints
    ckpt_dir = Path('./weights')
    ckpt_dir.mkdir(exist_ok=True)

    ckpt_path = ckpt_dir / f"{'SinSR_v1.pth' if model == 'SinSR' else 'resshift_' + task + '_s15_v1.pth'}"
    if not ckpt_path.exists():
        load_file_from_url(
            url=f"https://github.com/{'wyf0912/SinSR/releases/download/v1.0' if model == 'SinSR' else 'zsyOAOA/ResShift/releases/download/v2.0'}/{ckpt_path.name}",
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name
        )

    if model == 'ResShift':
        vqgan_path = ckpt_dir / 'autoencoder_vq_f4.pth'
        if not vqgan_path.exists():
            load_file_from_url(
                "https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
                model_dir=ckpt_dir,
                progress=True,
                file_name=vqgan_path.name
            )
        configs.autoencoder.ckpt_path = str(vqgan_path)

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = 15
    configs.diffusion.params.sf = 4
    return configs


sampler_dict = {"SinSR": None, "ResShift": None}
tokenizer, ocr_model = init_models()


def process_image(image,
                  enable_super_res=True,
                  enable_ocr=True,
                  super_res_model="SinSR",
                  ocr_type="ocr",
                  colab=False,
                  seed=12345):
    """Unified processing pipeline"""
    results = {
        "original_image": image,
        "enhanced_image": None,
        "ocr_text": None,
        "status": "Success"
    }

    try:
        # Save input image if needed
        if isinstance(image, np.ndarray):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                Image.fromarray(image).save(temp_file.name)
                image_path = temp_file.name
        else:
            image_path = image

        # Super Resolution
        if enable_super_res:
            configs = get_configs(super_res_model, colab)
            if sampler_dict[super_res_model] is None:
                sampler_dict[super_res_model] = Sampler(
                    configs,
                    chop_size=256,
                    chop_stride=224,
                    chop_bs=1,
                    use_fp16=True,
                    seed=seed
                )

            out_dir = Path('restored_output')
            out_dir.mkdir(exist_ok=True)

            sampler_dict[super_res_model].inference(
                image_path,
                out_dir,
                bs=1,
                noise_repeat=False,
                one_step=(super_res_model == "SinSR")
            )

            enhanced_path = out_dir / f"{Path(image_path).stem}.png"
            results["enhanced_image"] = util_image.imread(enhanced_path, chn="rgb", dtype="uint8")

            # Use enhanced image for OCR
            image_for_ocr = enhanced_path
        else:
            image_for_ocr = image_path

        # OCR
        if enable_ocr:
            results['ocr_text'] = ocr_model.chat(tokenizer, str(image_for_ocr), ocr_type='ocr')

            # inputs = tokenizer(image_for_ocr, return_tensors="pt", padding=True)
            # ocr_output = ocr_model.generate(**inputs)
            # results["ocr_text"] = tokenizer.decode(ocr_output[0], skip_special_tokens=True)
    except Exception as e:
        results["status"] = f"Error: {str(e)}"

    return (
        results["enhanced_image"] if results["enhanced_image"] is not None else results["original_image"],
        results["ocr_text"],
        results["status"]
    )


# Gradio Interface
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Enhanced Image Processing Pipeline
        ## Super Resolution and OCR Integration
        Upload an image to enhance its quality and/or extract text content.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="filepath", label="Input Image")
                with gr.Row():
                    enable_super_res = gr.Checkbox(label="Enable Super Resolution", value=True)
                    enable_ocr = gr.Checkbox(label="Enable OCR", value=True)

                with gr.Row():
                    super_res_model = gr.Dropdown(
                        choices=["SinSR", "ResShift"],
                        value="SinSR",
                        label="Super Resolution Model"
                    )
                    ocr_type = gr.Dropdown(
                        choices=["ocr", "format"],
                        value="ocr",
                        label="OCR Type"
                    )

                with gr.Row():
                    colab_mode = gr.Checkbox(label="Colab Mode", value=False)
                    seed = gr.Number(value=12345, label="Random Seed", precision=0)

                process_btn = gr.Button("Process Image", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(type="numpy", label="Processed Image")
                ocr_output = gr.Textbox(label="OCR Result", lines=5)
                status_output = gr.Textbox(label="Status")

        process_btn.click(
            fn=process_image,
            inputs=[
                input_image,
                enable_super_res,
                enable_ocr,
                super_res_model,
                ocr_type,
                colab_mode,
                seed
            ],
            outputs=[output_image, ocr_output, status_output]
        )

        gr.Markdown("""
        ### Notes:
        - Super Resolution enhances image quality
        - OCR extracts text from the image
        - You can use either or both features
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.queue(concurrency_count=1)
    demo.launch(share=True)
