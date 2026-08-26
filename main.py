import asyncio
import os
import time
import base64
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

import psutil
from PIL import Image, ImageChops
import numpy as np
import mss
import astrbot.api.message_components as Comp
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult, MessageChain
from astrbot.api.star import Context, Star
from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

# 兼容不同版本：尝试导入 ImagePart，失败则降级为纯文本
try:
    from astrbot.core.agent.message import TextPart, UserMessageSegment, ImagePart
except ImportError:
    from astrbot.core.agent.message import TextPart, UserMessageSegment
    ImagePart = None

# ------------------------------------------------------------------
# 跨平台 GPU 监控模块
# ------------------------------------------------------------------
class GPUManager:
    """自动检测并初始化显卡监控库"""
    def __init__(self):
        self.vendor = "NONE"
        self.nvml_handle = None
        self.amd_handle = None
        self.intel_dev = None

        self._try_init_nvidia()
        self._try_init_amd()
        self._try_init_intel()

    def _try_init_nvidia(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.vendor = "NVIDIA"
            logger.info("已检测到 NVIDIA GPU，使用 pynvml 监控")
        except Exception:
            pass

    def _try_init_amd(self):
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            devices = amdsmi.amdsmi_get_processor_handles()
            if len(devices) > 0:
                self.amd_handle = devices[0]
                self.vendor = "AMD"
                logger.info("已检测到 AMD GPU，使用 amdsmi 监控")
        except Exception:
            pass

    def _try_init_intel(self):
        try:
            import pyzes
            os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")
            self.intel_dev = True
            self.vendor = "INTEL"
            logger.info("已检测到 Intel GPU，使用 Level Zero Sysman 监控")
        except Exception:
            pass

    def get_gpu_utilization(self) -> Optional[float]:
        """返回 GPU 使用率百分比，失败返回 None"""
        if self.vendor == "NVIDIA" and self.nvml_handle:
            try:
                import pynvml
                return pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle).gpu
            except Exception:
                return None
        if self.vendor == "AMD" and self.amd_handle:
            try:
                import amdsmi
                return amdsmi.amdsmi_get_gpu_activity(self.amd_handle)["gfx_activity"]
            except Exception:
                return None
        if self.vendor == "INTEL" and self.intel_dev:
            return None
        return None


class ScreenMonitorPlugin(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.config = config
        self.plugin_name = self.name
        self.monitor_task: Optional[asyncio.Task] = None
        self._last_event_time = 0
        self._is_recording = False
        self._screenshots: List[Path] = []
        self._prev_screen = None
        self._storage_dir = Path(config.get("storage_dir", "")) if config.get("storage_dir") else Path(get_astrbot_plugin_data_path()) / self.plugin_name
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._finish_lock = asyncio.Lock()
        self._low_load_since = None
        self._process_absent_since = None
        self.gpu_manager = GPUManager()
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        self._pending_start_time = None

    async def terminate(self):
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        try:
            while True:
                try:
                    process_name = self.config.get("process_name", "")
                    process_names_config = self.config.get("process_name", [])
                    if isinstance(process_names_config, str):
                        process_names_config = [process_names_config]

                    is_process_running = False
                    cpu_percent = 0.0
                    gpu_percent = 0.0
                    high_load = False

                    if process_names_config:
                        for proc in psutil.process_iter(['name']):
                            try:
                                proc_name = proc.info['name']
                                if proc_name and any(proc_name.lower() == p.lower() for p in process_names_config):
                                    is_process_running = True
                                    break
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue
                    else:
                        cpu_percent = psutil.cpu_percent(interval=None)
                        gpu_percent = self.gpu_manager.get_gpu_utilization() if self.config.get("enable_gpu", False) else 0.0
                        # 解析 CPU 阈值区间
                        cpu_min, cpu_max = self._parse_range(
                            self.config.get("cpu_threshold", "50-70"), 
                            default_min=50, 
                            default_max=70
                        )
                        high_load = cpu_min <= cpu_percent <= cpu_max

                        if self.config.get("enable_gpu", False) and gpu_percent is not None:
                            gpu_min, gpu_max = self._parse_range(
                                self.config.get("gpu_threshold", "50-70"), 
                                default_min=50, 
                                default_max=70
                            )
                            high_load = high_load or (gpu_min <= gpu_percent <= gpu_max)

                    should_start = False
                    if process_name:
                        should_start = is_process_running
                    else:
                        screen_changed = False
                        if high_load:
                            screen = self._capture_screen()
                            if screen is not None:
                                if self._prev_screen is not None:
                                    diff_ratio = self._image_diff_ratio(self._prev_screen, screen)
                                    if diff_ratio >= self.config.get("screen_change_threshold", 5.0):
                                        screen_changed = True
                                self._prev_screen = screen
                        should_start = high_load and screen_changed

                    if self._is_recording:
                        end_condition = False
                        if process_name:
                            if not is_process_running:
                                if self._process_absent_since is None:
                                    self._process_absent_since = time.time()
                                elif time.time() - self._process_absent_since >= self.config.get("process_end_duration", 5):
                                    end_condition = True
                            else:
                                self._process_absent_since = None
                        else:
                            if not high_load:
                                if self._low_load_since is None:
                                    self._low_load_since = time.time()
                                elif time.time() - self._low_load_since >= self.config.get("low_load_duration", 10):
                                    end_condition = True
                            else:
                                self._low_load_since = None

                        if end_condition:
                            await self._finish_recording()
                            self._low_load_since = None
                            self._process_absent_since = None
                        else:
                            if self._screenshots:
                                last_shot = self._screenshots[-1]
                                if time.time() - last_shot.stat().st_mtime >= self.config.get("screenshot_interval", 5.0):
                                    self._take_screenshot()
                            else:
                                self._take_screenshot()
                    else:
                        # 非录制状态，处理启动条件
                        if should_start:
                            now = time.time()
                            # 先检查冷却时间
                            if now - self._last_event_time >= self.config.get("cooldown", 300):
                                # 冷却时间已过，开始计算“等待开始”时间
                                if self._pending_start_time is None:
                                    self._pending_start_time = now
                                    wait_time = self.config.get("process_start_duration", 5)
                                    logger.info(f"检测到潜在事件，等待 {wait_time} 秒后开始记录...")
                                elif now - self._pending_start_time >= self.config.get("process_start_duration", 5):
                                    # 等待时间已到，真正开始录制
                                    self._pending_start_time = None
                                    await self._start_recording()
                                    self._low_load_since = None
                                    self._process_absent_since = None
                                # else: 还在等待中，继续循环
                            else:
                                # 冷却时间未过，重置等待状态
                                self._pending_start_time = None
                        else:
                            # 条件不满足，重置等待状态
                            self._pending_start_time = None

                    await asyncio.sleep(self.config.get("check_interval", 2.0))
                except Exception as e:
                    logger.error(f"监控循环异常: {e}")
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("监控任务已取消")
            raise

    @staticmethod
    def _parse_resolution(value: str) -> Optional[tuple]:
        """
        解析分辨率预设字符串，返回 (宽, 高) 元组；若返回 None 则表示不缩放。
        """
        mapping = {
            "原始": None,
            "4k": (3840, 2160),
            "2k": (2560, 1440),
            "1080p": (1920, 1080),
            "720p": (1280, 720),
            "480p": (854, 480)
        }
        key = str(value).strip().lower()
        if key in mapping:
            return mapping[key]
        # 兼容手动输入形如 "1920x1080" 的格式
        if "x" in key:
            try:
                w, h = key.split("x")
                return (int(w), int(h))
            except (ValueError, IndexError):
                return None
        return None

    def _capture_screen(self) -> Optional[Image.Image]:
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                shot = sct.grab(monitor)
                return Image.frombytes("RGB", shot.size, shot.rgb)
        except Exception as e:
            logger.error(f"截图失败: {e}")
            return None

    def _image_diff_ratio(self, img1: Image.Image, img2: Image.Image) -> float:
        try:
            if img1.size != img2.size:
                return 100.0
            diff = ImageChops.difference(img1, img2)
            arr = np.array(diff)
            return np.count_nonzero(arr) / arr.size * 100
        except Exception:
            return 0.0

    def _take_screenshot(self):
        try:
            img = self._capture_screen()
            if img is None:
                return

            # 解析目标分辨率
            target_res = self._parse_resolution(self.config.get("screenshot_resolution", "原始"))
            if target_res is not None:
                target_width, target_height = target_res
                # 仅当原始尺寸大于目标时才缩放
                if img.width > target_width or img.height > target_height:
                    ratio = min(target_width / img.width, target_height / img.height)
                    new_width = int(img.width * ratio)
                    new_height = int(img.height * ratio)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    logger.info(f"截图已缩放: {img.width}x{img.height} -> {new_width}x{new_height}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            quality = self.config.get("screenshot_quality", 85)
            if 0 < quality < 100:
                path = self._storage_dir / f"event_{timestamp}.jpg"
                img.save(path, "JPEG", quality=quality)
            else:
                path = self._storage_dir / f"event_{timestamp}.png"
                img.save(path)
            self._screenshots.append(path)
            logger.info(f"已保存截图: {path}")
        except Exception as e:
            logger.error(f"保存截图失败: {e}")

    async def _start_recording(self):
        self._is_recording = True
        self._screenshots = []
        self._take_screenshot()
        logger.info("开始记录屏幕事件")

    def _clear_screenshots(self):
        for img in self._screenshots:
            try:
                img.unlink(missing_ok=True)
            except Exception:
                pass
        self._screenshots = []

    async def _finish_recording(self):
        if self._finish_lock.locked():
            return
        async with self._finish_lock:
            self._is_recording = False
            self._last_event_time = time.time()
            logger.info(f"结束记录，共 {len(self._screenshots)} 张截图")
            if not self._screenshots:
                return
            target_umo = self.config.get("target_umo", "")
            if not target_umo:
                logger.warning("未配置目标 UMO，无法发送消息")
                self._clear_screenshots()
                return
            if target_umo.count(':') < 2:
                logger.warning("目标 UMO 格式不正确，应为 platform:message_type:session_id，忽略发送")
                self._clear_screenshots()
                return
            try:
                max_images = int(self.config.get("max_images", 3))
                if len(self._screenshots) > max_images:
                    indices = np.linspace(0, len(self._screenshots) - 1, max_images).astype(int)
                    selected = [self._screenshots[i] for i in indices]
                else:
                    selected = self._screenshots

                logger.info(f"选中的图片路径: {[str(p) for p in selected]}")

                # 是否每张图片单独生成描述并立即发送，默认 True
                send_each_image_separately = self.config.get("send_each_image_separately", True)

                if send_each_image_separately:
                    # 逐张处理并立即发送
                    for img_path in selected:
                        if not img_path.exists():
                            logger.warning(f"截图文件不存在，跳过: {img_path}")
                            continue
                        # 单张图片调用 LLM 生成描述
                        description = await self._generate_description([img_path])
                        
                        # 构建这条图片对应的消息链
                        chain = MessageChain()
                        if self.config.get("send_images", True):
                            chain.file_image(str(img_path))
                        chain.message(description)
                        
                        # 立即发送
                        await self.context.send_message(target_umo, chain)
                        logger.info(f"已发送图片 {img_path.name} 及其描述")
                        
                        # 可选：适当等待，避免发送过快
                        # await asyncio.sleep(0.5)
                else:
                    # 原有逻辑：合并所有图片后一次性发送
                    chain = MessageChain()
                    if self.config.get("send_images", True):
                        for img_path in selected:
                            if not img_path.exists():
                                logger.warning(f"截图文件不存在，跳过: {img_path}")
                                continue
                            chain.file_image(str(img_path))
                    description = await self._generate_description(selected)
                    chain.message(description)
                    if len(chain.chain) == 0:
                        chain = MessageChain().message("（没有可发送的内容）")
                    await self.context.send_message(target_umo, chain)
                    logger.info("事件描述已发送")

            except Exception as e:
                logger.error(f"发送消息失败: {e}")
            finally:
                self._clear_screenshots()

    @staticmethod
    def _parse_range(value, default_min: float, default_max: float) -> tuple:
        """
        解析区间字符串，支持逗号/空格/~/-分隔。
        返回 (min, max)，解析失败时返回默认值。
        """
        if value is None:
            return default_min, default_max
        if isinstance(value, (int, float)):
            # 如果传的是单个数字，则默认区间为 [value, value]
            return float(value), float(value)
        if isinstance(value, (list, tuple)):
            # 如果传的是列表，取前两个元素
            vals = [float(x) for x in value[:2]]
            if len(vals) == 1:
                return vals[0], vals[0]
            return min(vals[0], vals[1]), max(vals[0], vals[1])
        # 字符串处理
        text = str(value).strip()
        # 支持 ~ 或 - 分隔，但注意负号可能出现在数值内（如 -10~50）
        parts = None
        for sep in ['~', '-', ',', ' ', '\t', '\n']:
            if sep in text:
                parts = [p.strip() for p in text.split(sep) if p.strip() != '']
                if len(parts) >= 2:
                    break
        if parts and len(parts) >= 2:
            try:
                first = float(parts[0])
                second = float(parts[1])
                return min(first, second), max(first, second)
            except ValueError:
                pass
        # 如果只有一个数字
        try:
            single = float(text)
            return single, single
        except ValueError:
            return default_min, default_max

    async def _generate_description(self, image_paths: List[Path]) -> str:
        provider_id = self.config.get("provider_id", "")
        if not provider_id:
            try:
                provider_id = await self.context.get_current_chat_provider_id(None)
            except Exception:
                provider_id = ""
        if not provider_id:
            logger.error("无法确定 LLM Provider，请配置 provider_id 或确保 AstrBot 已配置默认 Provider")
            return "（未能生成描述，因为未配置 LLM Provider）"

        persona_id = self.config.get("persona", "")
        persona_prompt = ""
        if persona_id:
            try:
                persona_obj = await self.context.persona_manager.get_persona(persona_id)
                if persona_obj:
                    persona_prompt = persona_obj.system_prompt
            except Exception as e:
                logger.error(f"获取人格 {persona_id} 失败: {e}")
                persona_prompt = ""
        if not persona_prompt:
            persona_prompt = self.config.get("persona", "你是一个观察者，用幽默的口吻描述发生的事情。")

        logger.info(f"provider_id: {provider_id}, 图片数量: {len(image_paths)}")
        logger.info(f"ImagePart is None: {ImagePart is None}")

        # 如果 ImagePart 不可用，直接走直连 API（兼容 Ollama 和云端）
        if ImagePart is None:
            logger.info("使用直连 API 识图")
            return await self._generate_description_direct(image_paths, persona_prompt)

        # 以下为 ImagePart 可用时的逻辑（保留）
        contexts = None
        if ImagePart is not None:
            try:
                content = []
                for img_path in image_paths:
                    mime_type = "image/jpeg" if img_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                    content.append(ImagePart(image=str(img_path), mime_type=mime_type))
                    logger.info(f"ImagePart 添加成功: {img_path}")
                prompt_template = self.config.get("prompt_template", "")
                if not prompt_template:
                    prompt_template = "请仔细观察这些连续截图，用符合以下人设的语气说出来。\n【人设】{persona}"
                prompt_text = prompt_template.replace("{persona}", persona_prompt)
                content.append(TextPart(text=prompt_text))
                user_msg = UserMessageSegment(content=content)
                contexts = [user_msg]
                logger.info(f"UserMessageSegment 构建成功，内容类型: {type(content[0])}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.warning(f"多模态构建失败，降级为纯文本: {e}")
                contexts = None

        try:
            system_prompt = "你是一个观察者，根据上下文猜测发生了什么，并用指定人设的语气描述"
            if contexts is not None:
                logger.info("使用多模态上下文调用 LLM")
                resp = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt="",
                    contexts=contexts,
                    system_prompt=system_prompt
                )
            else:
                logger.warning("使用纯文本上下文调用 LLM")
                text_prompt = (
                    f"用户刚刚玩了游戏或制作完了一个很大的工程，屏幕发生了明显变化。"
                    f"请根据以下描述（无图片）结合人设简要说明可能发生的情况：\n"
                    f"【人设】{persona_prompt}"
                )
                resp = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=text_prompt,
                    system_prompt=system_prompt
                )
            if resp and resp.completion_text:
                return resp.completion_text.strip()
            return "（未获取到描述）"
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return f"（描述生成失败：{e}）"

    async def _generate_description_direct(self, image_paths: List[Path], persona_prompt: str) -> str:
        import base64
        import httpx

        provider_id = self.config.get("provider_id", "")
        api_base = self.config.get("ollama_api_url", "").rstrip("/")
        api_key = self.config.get("api_key", "")

        # 尝试从 Provider 管理器获取配置
        try:
            provider_manager = self.context.provider_manager
            providers = provider_manager.get_providers() if hasattr(provider_manager, 'get_providers') else []
            for p in providers:
                if p.id == provider_id:
                    api_base = getattr(p, 'api_base', getattr(p, 'base_url', api_base))
                    api_key = getattr(p, 'api_key', api_key)
                    break
        except Exception as e:
            logger.warning(f"获取 Provider 配置失败: {e}")

        # 判断是否为 Ollama 本地
        is_ollama = "ollama" in provider_id.lower() or "127.0.0.1" in api_base or "localhost" in api_base

        model_name = provider_id.replace("ollama/", "")

        prompt_template = self.config.get("prompt_template", "")
        if not prompt_template:
            prompt_template = "请仔细观察这些连续截图，用符合以下人设的语气说出来。\n【人设】{persona}"
        prompt_text = prompt_template.replace("{persona}", persona_prompt)

        # 构建内容
        content_parts = []
        images = []  # 用于 Ollama 格式
        for img_path in image_paths:
            try:
                img_bytes = img_path.read_bytes()
                b64 = base64.b64encode(img_bytes).decode()
                if is_ollama:
                    images.append(b64)
                else:
                    data_uri = f"data:image/png;base64,{b64}"
                    content_parts.append({"type": "image_url", "image_url": {"url": data_uri}})
                logger.info(f"已编码图片: {img_path}")
            except Exception as e:
                logger.error(f"图片编码失败 {img_path}: {e}")

        if is_ollama:
            content_parts.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": prompt_text, "images": images}]
        else:
            content_parts.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content_parts}]

        logger.info(f"请求模型: {model_name}, 消息结构: {messages}")  # 注意：此日志可能泄露 base64，建议注释掉

        max_retries = 3  # 最大重试次数
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    if is_ollama:
                        endpoint = f"{api_base}/api/chat" if not api_base.endswith("/api/chat") else api_base
                        headers = {}
                        body = {"model": model_name, "messages": messages, "stream": False}
                    else:
                        if not api_base.endswith("/v1"):
                            api_base += "/v1"
                        endpoint = f"{api_base}/chat/completions"
                        headers = {"Authorization": f"Bearer {api_key}"}
                        body = {"model": model_name, "messages": messages}

                    resp = await client.post(endpoint, json=body, headers=headers)
                    # 记录状态码和响应体（截断，防止日志过长）
                    logger.debug(f"API 状态码: {resp.status_code}, 响应内容前200字符: {resp.text[:200]}")

                    if resp.status_code != 200:
                        error_msg = resp.text if resp.text else "响应体为空"
                        logger.warning(f"第 {attempt+1} 次请求失败: 状态码 {resp.status_code}, 错误信息: {error_msg[:200]}")
                        # 非 200 视为失败，重试
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                            continue
                        else:
                            return f"（API错误：{resp.status_code} - {error_msg[:200]}）"

                    data = resp.json()
                    if is_ollama:
                        content = data.get("message", {}).get("content", "")
                    else:
                        content = data["choices"][0]["message"]["content"]

                    logger.info(f"直连 API 返回：{content[:100]}")
                    return content.strip() if content else "（未获取到描述）"

            except httpx.TimeoutException as e:
                logger.warning(f"第 {attempt+1} 次请求超时: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    return "（直连 API 超时，请检查 Ollama 是否响应）"
            except httpx.HTTPStatusError as e:
                response_text = e.response.text if e.response else "无响应"
                logger.warning(f"第 {attempt+1} 次请求 HTTP 错误: {e}, 响应内容: {response_text[:200]}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    return f"（直连 API HTTP 错误：{e} - {response_text[:200]}）"
            except Exception as e:
                logger.warning(f"第 {attempt+1} 次请求异常: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                else:
                    return f"（直连 API 失败：{type(e).__name__} - {e}）"

        # 理论上不会走到这里，但为了安全返回一个默认值
        return "（直连 API 多次尝试后仍然失败）"


    # ---------- 指令 ----------
    @filter.command("am_status")
    async def am_status(self, event: AstrMessageEvent):
        """查看监控状态"""
        recording = "正在记录" if self._is_recording else "空闲"
        gpu_vendor = self.gpu_manager.vendor if self.config.get("enable_gpu", False) else "未启用"
        yield event.plain_result(f"当前状态: {recording}\n已记录截图数: {len(self._screenshots)}\nGPU监控: {gpu_vendor}")

    @filter.command("umo")
    async def umo(self, event: AstrMessageEvent):
        '''获取当前会话的 UMO'''
        yield event.plain_result(f"当前会话 UMO 为: {event.unified_msg_origin}")

    @filter.command("am_start")
    async def am_start(self, event: AstrMessageEvent):
        """手动触发一次记录：每5秒截一张图，共4次，然后发送给LLM"""
        asyncio.create_task(self.am_test())
        yield event.plain_result("已开始手动录制：每5秒截一张图，共4次，完成后发送。")

    async def am_test(self):
        self._is_recording = True
        self._screenshots = []
        for _ in range(4):
            self._take_screenshot()
            await asyncio.sleep(5)
        self._is_recording = False
        await self._finish_recording()

    @filter.command("am_clear")
    async def am_clear(self, event: AstrMessageEvent):
        """清除当前会话的记忆，防止旧话题干扰"""
        try:
            umo = event.unified_msg_origin
            if not umo:
                yield event.plain_result("无法获取当前会话标识。")
                return
            conv_mgr = self.context.conversation_manager
            new_cid = await conv_mgr.new_conversation(umo)
            yield event.plain_result(
                f"✅ 已为新模型清空对话历史 (新对话ID: {new_cid})。\n"
                "⚠️ 为防止其他模块误读旧消息，建议您手动清除平台历史消息缓存：\n"
                "在 AstrBot 数据目录中执行：\n"
                "sqlite3 data/data_v4.db \"DELETE FROM platform_message_history WHERE unified_msg_origin='{umo}';\"\n"
                "（如果字段名不同，请查看实际表结构）"
            )
            logger.info(f"已清除会话记忆: {umo}")
        except Exception as e:
            logger.error(f"清除记忆失败: {e}")
            yield event.plain_result(f"清除记忆失败: {e}")