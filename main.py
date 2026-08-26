import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import psutil
from PIL import Image, ImageChops
import numpy as np
import mss
import astrbot.api.message_components as Comp
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star
from astrbot.api import logger
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path
from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult, MessageChain

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
        self.vendor = "NONE"  # NVIDIA / AMD / INTEL / NONE
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
        # 基于 Level Zero Sysman (pyzes) 的简化实现
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
            # Intel 需要更复杂 Level Zero 调用，返回占位示例
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
        self._low_counter = 0
        self._screenshots: List[Path] = []
        self._prev_screen = None
        self._storage_dir = Path(config.get("storage_dir", "")) if config.get("storage_dir") else Path(get_astrbot_plugin_data_path()) / self.plugin_name
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self.gpu_manager = GPUManager()

        # 直接启动监控任务
        self.monitor_task = asyncio.create_task(self._monitor_loop())

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
                    cpu_percent = psutil.cpu_percent(interval=None)
                    gpu_percent = self.gpu_manager.get_gpu_utilization() if self.config.get("enable_gpu", False) else 0.0

                    high_load = cpu_percent >= self.config.get("cpu_threshold", 80)
                    if self.config.get("enable_gpu", False) and gpu_percent is not None:
                        high_load = high_load or gpu_percent >= self.config.get("gpu_threshold", 80)

                    screen_changed = False
                    if high_load:
                        screen = self._capture_screen()
                        if screen is not None:
                            if self._prev_screen is not None:
                                diff_ratio = self._image_diff_ratio(self._prev_screen, screen)
                                if diff_ratio >= self.config.get("screen_change_threshold", 5.0):
                                    screen_changed = True
                            self._prev_screen = screen

                    if self._is_recording:
                        if not high_load:
                            self._low_counter += 1
                            if self._low_counter >= 2:
                                await self._finish_recording()
                                self._low_counter = 0
                        else:
                            self._low_counter = 0
                            if self._screenshots:
                                last_shot = self._screenshots[-1]
                                if time.time() - last_shot.stat().st_mtime >= self.config.get("screenshot_interval", 5.0):
                                    self._take_screenshot()
                            else:
                                self._take_screenshot()
                    else:
                        now = time.time()
                        if now - self._last_event_time >= self.config.get("cooldown", 300):
                            if high_load and screen_changed:
                                await self._start_recording()

                    await asyncio.sleep(self.config.get("check_interval", 2.0))
                except Exception as e:
                    logger.error(f"监控循环异常: {e}")
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            logger.info("监控任务已取消")
            raise

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
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

    async def _finish_recording(self):
        self._is_recording = False
        self._last_event_time = time.time()
        logger.info(f"结束记录，共 {len(self._screenshots)} 张截图")

        if not self._screenshots:
            return

        target_umo = self.config.get("target_umo", "")
        if not target_umo:
            logger.warning("未配置目标 UMO，无法发送消息")
            return

        # 校验 UMO 格式，避免发送失败
        if target_umo.count(':') < 2:
            logger.warning("目标 UMO 格式不正确，应为 platform:message_type:session_id，忽略发送")
            return

        try:
            max_images = int(self.config.get("max_images", 3))
            if len(self._screenshots) > max_images:
                indices = np.linspace(0, len(self._screenshots) - 1, max_images).astype(int)
                selected = [self._screenshots[i] for i in indices]
            else:
                selected = self._screenshots

            description = await self._generate_description(selected)

            # 使用 MessageChain 构建消息
            chain = MessageChain()
            if self.config.get("send_images", True):
                for img_path in selected:
                    chain.file_image(str(img_path))  # 或使用 chain.image(...) 根据版本选择
            chain.message(description)

            await self.context.send_message(target_umo, chain)
            logger.info("事件描述已发送")
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
        finally:
            # 清理临时截图
            for img in self._screenshots:
                try:
                    img.unlink(missing_ok=True)
                except Exception:
                    pass
            self._screenshots = []

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

        # 获取人格设定
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
            persona_prompt = self.config.get("persona", "我是一个观察者，用幽默的口吻描述发生的事情。")

        # 构建上下文（多模态或纯文本）
        contexts = None
        if ImagePart is not None:
            try:
                content = []
                for img_path in image_paths:
                    content.append(ImagePart(image=str(img_path)))
                
                # 使用用户自定义提示词模板，支持 {persona} 占位符
                prompt_template = self.config.get("prompt_template", "")
                if not prompt_template:
                    prompt_template = "请仔细观察这些连续截图。这通常是用户玩游戏（如射击游戏等）、进行高强度创作或运行大型软件时的画面。请基于画面内容，详细描述发生的动作和结果（例如：击杀了敌人、完成了操作、进度条走完等）。请用符合以下人设的语气说出来。\n【人设】{persona}"
                prompt_text = prompt_template.replace("{persona}", persona_prompt)
                content.append(TextPart(text=prompt_text))
                
                user_msg = UserMessageSegment(content=content)
                contexts = [user_msg]
            except Exception as e:
                logger.warning(f"多模态构建失败，降级为纯文本: {e}")
                contexts = None

        try:
            if contexts is not None:
                # 多模态调用（图片 + 文本）
                resp = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt="",
                    contexts=contexts,
                    system_prompt="你是一个观察者，从截图序列中推断发生了什么，并用指定人设的语气描述。"
                )
            else:
                # 降级：纯文本描述（无图片信息）
                text_prompt = (
                    f"我刚刚经历了一次高负载事件（CPU/GPU 飙升），屏幕发生了明显变化。"
                    f"请根据以下描述（无图片）结合人设简要说明可能发生的情况：\n"
                    f"【人设】{persona_prompt}"
                )
                resp = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=text_prompt,
                    system_prompt="你是一个观察者，根据上下文猜测发生了什么，并用指定人设的语气描述。"
                )
            if resp and resp.completion_text:
                return resp.completion_text.strip()
            return "（未获取到描述）"
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return f"（描述生成失败：{e}）"

    # ---------- 指令 ----------
    @filter.command("screenmonitor_status")
    async def status(self, event: AstrMessageEvent):
        """查看监控状态"""
        recording = "正在记录" if self._is_recording else "空闲"
        gpu_vendor = self.gpu_manager.vendor if self.config.get("enable_gpu", False) else "未启用"
        yield event.plain_result(f"当前状态: {recording}\n已记录截图数: {len(self._screenshots)}\nGPU监控: {gpu_vendor}")

    @filter.command("get_umo")
    async def get_umo(self, event: AstrMessageEvent):
        '''获取当前会话的 UMO'''
        yield event.plain_result(f"当前会话 UMO 为: {event.unified_msg_origin}")

    @filter.command("screenmonitor_trigger")
    async def manual_trigger(self, event: AstrMessageEvent):
        """手动触发一次记录（用于测试）"""
        await self._start_recording()
        yield event.plain_result("手动触发记录，等待结束...")