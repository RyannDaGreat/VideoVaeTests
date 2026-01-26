import rp
import shlex

download_dir = rp.get_path_parent(__file__)
local_download_dir = "/models/LTX2" #For Eyeline computers, else make it = download_dir

model_urls = [
    "https://huggingface.co/" + x + ".safetensors"
    for x in [
        "Kijai/LTXV2_comfy/resolve/main/loras/ltx-2-19b-distilled-lora-resized_dynamic_fro095_avg_rank_242_bf16",
        "Lightricks/LTX-2-19b-IC-LoRA-Canny-Control/resolve/main/ltx-2-19b-ic-lora-canny-control",
        "Lightricks/LTX-2-19b-IC-LoRA-Depth-Control/resolve/main/ltx-2-19b-ic-lora-depth-control",
        "Lightricks/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer",
        "Lightricks/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control",
        "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in",
        "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left",
        "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out",
        "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right",
        "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down",
        "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up",
        "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static",
        "Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp4",
        "Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8",
        "Lightricks/LTX-2/resolve/main/ltx-2-19b-dev",
        "Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-fp8",
        "Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384",
        "Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled",
        "Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0",
        "Lightricks/LTX-2/resolve/main/ltx-2-temporal-upscaler-x2-1.0",
    ]
]

def download_from_web():
    with rp.SetCurrentDirectoryTemporarily(download_dir):
        for url in rp.eta(model_urls, "Downloading Models"):
            # print(url)
            # print()
            rp.download_url(url, skip_existing=True, show_progress=True)
            # print()

def download_to_local():
    download_from_web()
    if download_dir == local_download_dir:
        return
    command = f'rclone copy --progress --transfers 32 {shlex.quote(download_dir)} {shlex.quote(local_download_dir)}'
    rp.r._run_sys_command(command)

if __name__=='__main__':
    download_to_local()
