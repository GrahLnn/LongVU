import tempfile
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import filetype
import numpy as np
import torch
from decord import VideoReader, cpu
from PIL import Image, ImageSequence
from transformers import BitsAndBytesConfig

from .builder import load_pretrained_model
from .constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from .conversation import SeparatorStyle, conv_templates
from .mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from .utils import check_model


class MediaType(Enum):
    VIDEO = "video"
    IMAGE = "image"
    UNKNOWN = "unknown"


class ModelType(Enum):
    LLAMA = "llama"
    QWEN = "qwen"


@dataclass
class ModelInstance:
    tokenizer: Optional[object] = None
    model: Optional[object] = None
    image_processor: Optional[object] = None
    context_len: Optional[int] = None
    current_model_type: Optional[ModelType] = None
    current_media_type: Optional[MediaType] = None


class ModelManager:
    _instance: Optional[ModelInstance] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> ModelInstance:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ModelInstance()
        return cls._instance

    @classmethod
    def _load_model(
        cls, model_type: ModelType, media_type: MediaType, quant: str = "8bit"
    ) -> None:
        instance = cls.get_instance()
        # 只有在模型类型或媒体类型改变时才重新加载
        if (
            instance.model is None
            or instance.current_model_type != model_type
            or instance.current_media_type != media_type
        ):
            model_suffix = "v" if media_type == MediaType.VIDEO else "img"
            match model_type:
                case ModelType.LLAMA:
                    model_name = "cambrian_llama"
                    model_path = check_model(f"llama{model_suffix}")
                case ModelType.QWEN:
                    model_name = "cambrian_qwen"
                    model_path = check_model(f"qwen{model_suffix}")
            match quant:
                case "4bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                case "8bit":
                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True
                    )
                case _:
                    quantization_config = None
            (
                instance.tokenizer,
                instance.model,
                instance.image_processor,
                instance.context_len,
            ) = load_pretrained_model(
                model_path, None, model_name, quantization_config=quantization_config
            )

            instance.model.eval()
            instance.current_model_type = model_type
            instance.current_media_type = media_type


def convert_gif_to_video(gif_path):
    # 创建临时视频文件
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_path = temp_video.name

    # 读取GIF
    gif = Image.open(gif_path)

    # 获取GIF尺寸
    width, height = gif.size

    # 设置视频编码器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_path, fourcc, 10.0, (width, height))

    # 将每一帧写入视频
    for frame in ImageSequence.Iterator(gif):
        # 转换为RGB并转为numpy数组
        frame_array = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
        out.write(frame_array)

    out.release()
    return temp_path


def infer(
    media: str,
    prompt: str = "Describe this media in detail.",
    model_type: str = "llama",
    quant: str = "8bit",
):
    """
    Model type: llama, qwen
    Quant mode: 4bit, 8bit
    """
    try:
        model_type = ModelType(model_type)
    except ValueError:
        raise ValueError(
            f"Invalid model type: {model_type}. Must be one of {[e.value for e in ModelType]}"
        )

    # 检查媒体类型
    kind = filetype.guess(media)
    if kind is None:
        raise ValueError("Cannot determine file type")

    mime = kind.mime
    media_type = None
    match mime.split("/")[0]:
        case "video":
            media_type = MediaType.VIDEO
        case "image":
            if mime == "image/gif":
                media = convert_gif_to_video(media)
                media_type = MediaType.VIDEO
            else:
                media_type = MediaType.IMAGE
        case _:
            media_type = MediaType.UNKNOWN

    if media_type == MediaType.UNKNOWN:
        raise ValueError(f"Unsupported media type: {mime}")

    # 加载模型
    ModelManager._load_model(model_type, media_type, quant)
    instance = ModelManager.get_instance()

    # 获取对话模板
    templete = "llama3_2" if model_type == ModelType.LLAMA else "qwen"

    # 统一处理媒体输入
    if media_type == MediaType.IMAGE:
        image = Image.open(media)
        media_array = [np.array(image)]
    else:  # VIDEO
        vr = VideoReader(media, ctx=cpu(0), num_threads=1)
        fps = float(vr.get_avg_fps())

        # 检查显存大小
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3  # 转换为GB

        # 仅对<24GB显存的设备限制帧数
        if vram < 24:
            num_frames = 1000 if len(vr) > 1000 else len(vr)
        else:
            num_frames = len(vr)

        frame_indices = np.array([i for i in range(0, num_frames, round(fps))])
        media_array = [vr[i].asnumpy() for i in frame_indices]

    image_sizes = [media_array[0].shape[:2]]
    media_array = process_images(
        media_array, instance.image_processor, instance.model.config
    )
    media_array = [item.unsqueeze(0) for item in media_array]

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = conv_templates[templete].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    whole_prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(
            whole_prompt, instance.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(instance.model.device)
    )

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(
        keywords, instance.tokenizer, input_ids
    )

    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output_ids = instance.model.generate(
            input_ids,
            images=media_array,
            attention_mask=attention_mask,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=4096,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    return instance.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
        0
    ].strip()
