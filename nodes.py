import os, sys, json, uuid
import torch
import torchaudio

from .stable_audio_tools.interface.gradio import load_model
from .stable_audio_tools.interface.interfaces.diffusion_cond import init_model_info, generate_cond

def get_comfyui_root():
    main_module = sys.modules.get('__main__')
    if main_module and hasattr(main_module, '__file__'):
        main_path = os.path.abspath(main_module.__file__)
        root_dir = os.path.dirname(main_path)
        return root_dir
    return None

class LoadStableAudioModel:

    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_filename": ("STRING", {"default": "model.safetensors"}),
            }
        }

    RETURN_TYPES = ("SAOMODEL",)
    RETURN_NAMES = ("audio_model",)
    FUNCTION = "generate"
    OUTPUT_NODE = True

    CATEGORY = "fg/StableAudio"

    def generate(self, model_filename):
        comfyui_root = get_comfyui_root()
        root_path = (comfyui_root or "").replace('/custom_nodes/ComfyUI-StableAudioFG', '')
        checkpoint_path = f"{root_path}/models/checkpoints/stable-audio"

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            with open(f"{root_path}/custom_nodes/ComfyUI-StableAudioFG/models/stable-audio/model_config.json") as f:
                model_config = json.load(f)

            # 模型加载（仅一次，在load_model或别的地方不要使用全局变量，否则无法及时清除显存）
            model, model_config = load_model(
                model_config,
                f"{checkpoint_path}/{model_filename}",
                pretrained_name=None,
                pretransform_ckpt_path=None,
                model_half=True,
                device=device
            )
            model_half = True

            # 方便显存清除？
            self.model = model

            init_model_info(model_config, model, model_half)
            return (model,)
        except Exception as e:
            print(f"load model error: {str(e)}")
        finally:
            self.model = None


class StableAudioFG:
    def __init__(self):
        self.model = None
        self.model_dir = None
        self.result_txt = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_model": ("SAOMODEL", {"forceInput": True}),
                "prompt": ("STRING",),
                "negative_prompt": ("STRING", {"default": "noise"}),
                "seed": ("INT", {"default": 0}),
                "steps": ("INT", {"default": 50, "min": 15, "max": 150}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 10.0, "step": 1.0}),
                "seconds_total": ("INT", {"default": 10, "min": 1, "max": 45}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "audio_path")
    FUNCTION = "generate"
    CATEGORY = "fg/StableAudio"
    OUTPUT_NODE = True

    def generate(self, audio_model, prompt, negative_prompt, seed, steps, cfg_scale, seconds_total):
        comfyui_root = get_comfyui_root()
        root_path = (comfyui_root or "").replace('/custom_nodes/ComfyUI-StableAudioFG', '')

        # 生成唯一文件名避免冲突
        unique_id = uuid.uuid4().hex
        base_file_name = f"audio_{unique_id}"
        output_directory = f"{root_path}/output"
        os.makedirs(output_directory, exist_ok=True)
        file_name = os.path.join(output_directory, base_file_name)

        try:
            # 生成音频文件
            generate_cond(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seconds_start=0,
                seconds_total=seconds_total,
                cfg_scale=cfg_scale,
                steps=steps,
                preview_every=None,
                seed=-1,
                sampler_type="dpmpp-3m-sde",
                sigma_min=0.01,
                sigma_max=100,
                rho=1.0,
                cfg_interval_min=0.0,
                cfg_interval_max=1.0,
                cfg_rescale=0.0,
                file_format="wav",
                file_naming=file_name,
                cut_to_seconds_total=True,
                # init_audio=None,
                # init_noise_level=0.1,
                # mask_maskstart=10,
                # mask_maskend=47,
                # inpaint_audio=None,
                # batch_size=1,
                model_in=audio_model,
                # sample_size_in=2097152,
                # sample_rate_in=44100,
                # model_type_in="diffusion_cond",
                # model_half_in=True
            )

            # 查找生成的音频文件（处理时间戳后缀）
            if os.path.exists(file_name+'.wav'):
                audio_path = file_name+'.wav'
                # print(f'音频路径--{audio_path}')

                # 加载音频为张量
                waveform, sample_rate = torchaudio.load(audio_path)
                
                # 返回音频张量、采样率和文件路径
                #return ((waveform, sample_rate), audio_path,)
                
                # 将(waveform, sample_rate)转换为目标字典格式
                audio_data = {
                    'waveform': waveform.unsqueeze(0),  # 增加批次维度
                    'sample_rate': sample_rate
                }
                
                # 返回新格式的数据
                return (audio_data, audio_path,)
        except Exception as e:
            print(f"Audio generation failed: {str(e)}")
        # finally:
        #     if torch.cuda.is_available():
        #         print(f'清空显存')
        #         audio_model = None
        #         torch.cuda.empty_cache()
        return (None, "",)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "LoadStableAudioModel": LoadStableAudioModel,
    "StableAudioFG": StableAudioFG,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadStableAudioModel": "Load Stable Audio Model",
    "StableAudioFG": "Stable Audio Do"
}
