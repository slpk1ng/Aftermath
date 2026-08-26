# 余波~ ( Aftermath ) v0.1.0

当你打完游戏或做完某个大项目时，bot会发送符合你设置的人设的句子来复盘并回应你！监控 CPU/GPU 占用和屏幕变化，当占用超过阈值且屏幕有明显变化时截图，占用下降后通过模型生成复盘并主动发送到指定会话。

---

## ✨ 主要特性

- 全平台 CPU 监控，并尝试自动检测并适配 NVIDIA / AMD / Intel GPU（需额外安装对应库）。
- 基于像素差异比例判断“明显的屏幕变化”，避免细微抖动误触发。
- 在满足触发条件时进行事件录制（期间按间隔截图），负载下降后自动结束。
- 可配置冷却时间，避免频繁重复触发。
- 支持调用 AstrBot 配置的 LLM（可选多模态）生成事件描述，结合 WebUI 中的人格设定（Persona）输出拟人化文本。
- 可将截图（可选）和文字描述推送到目标 UMO（目标会话）。
- 提供便捷调试指令：/get_umo、/screenmonitor_status、/screenmonitor_trigger。

---

## 📦 先决条件

- 运行 AstrBot 的主机需具有图形界面（用于截图）。Linux 环境需有 X11 或 Wayland。
- Python 3.8+（与宿主 AstrBot 兼容的 Python 版本）。

## 安装步骤

1. 将插件目录放入 AstrBot 插件目录：
   - 将 `astrbot_plugin_screen_monitor` 文件夹复制到 AstrBot 的 `data/plugins/` 目录。

2. 安装 Python 依赖：

```bash
pip install -r data/plugins/astrbot_plugin_screen_monitor/requirements.txt
```

如果希望手动安装或调试，常用依赖如下：

```bash
pip install psutil mss Pillow numpy
```

3. （可选）GPU 监控依赖：
- NVIDIA：pip install nvidia-ml-py
- AMD：pip install pyamdsmi
- Intel（不保证在所有系统可用）：pip install pyzes 并设置环境变量 ZES_ENABLE_SYSMAN=1

4. 在 AstrBot WebUI 的插件管理页中重载该插件或重启 AstrBot。

---

## ✅ 配置说明（WebUI 插件配置）

建议在 AstrBot 的插件配置界面填写以下字段。下面以表格形式列出（在 WebUI 中字段通常以 JSON schema 展示）：

| 配置项 | 类型 | 默认 | 说明 |
|---|---:|---:|---|
| cpu_threshold | float | 80.0 | CPU 占用率阈值（%），超过后开始检测屏幕变化 |
| enable_gpu | bool | false | 是否启用 GPU 监控（需要额外安装对应库） |
| gpu_threshold | float | 80.0 | GPU 占用率阈值（%），仅在 enable_gpu 为 true 时生效 |
| screen_change_threshold | float | 5.0 | 屏幕像素差异比例阈值（0-100），超过视为“明显变化” |
| cooldown | float | 300.0 | 事件冷却时间（秒），一次事件结束后需等待此时间才能再次触发 |
| check_interval | float | 2.0 | 资源检测间隔（秒），数值越小响应越快但消耗更多资源 |
| screenshot_interval | float | 5.0 | 事件期间截图间隔（秒） |
| target_umo | string | "" | 目标会话 UMO（例如 aiocqhttp:GroupMessage:123456789），留空则不发送 |
| persona | string | "" | 在 WebUI 中选择的人格 ID；插件会读取人格的 system_prompt 作为描述风格 |
| provider_id | string | "" | LLM Provider ID ，留空使用全局默认模型 |
| send_images | bool | true | 是否随消息发送截图 |
| max_images | int | 3 | 最多发送的截图数（事件期间均匀抽取） |
| storage_dir | string | "" | 截图保存目录（绝对路径），留空则使用插件数据目录 |

条件显示说明：gpu_threshold 仅在 enable_gpu 开启后显示；max_images 仅在 send_images 开启后显示。

---

## 🛠 指令（Commands）

- /get_umo — 获取当前会话的 UMO（用于配置 target_umo）。
- /screenmonitor_status — 查看当前监控状态、启用的设备类型、已记录截图数等信息。
- /screenmonitor_trigger — 手动触发一次录制事件（用于测试）。

---

## 使用示例（典型工作流）

1. 在 WebUI 中配置触发阈值与目标 UMO（或先留空以仅本地测试）。
2. 启动或重载插件，等待监控开始。
3. 当 CPU/GPU 高负载且屏幕发生明显变化时，插件会录制并在事件结束后生成摘要并发送到 target_umo（若已配置）。
4. 若使用 Persona 和 LLM，会将生成的文字描述附带图片一起发送（若模型支持多模态）。

---

## 故障排查与注意事项

- 截图失败：确认宿主有图形环境（Windows 桌面、macOS、Linux + X11/Wayland）。
- GPU 未检测或数据为 0：请检查是否安装了对应显卡的监控库（见上文），并确保权限允许访问监控接口。Intel 的 pyzes 在部分系统/驱动上不可用。
- LLM 生成失败：确认 AstrBot 全局已配置有效的聊天模型（provider），并确认模型是否支持图片输入；若不支持，可将 send_images 设为 false。
- 日志：请查看 AstrBot 的插件日志以获取详细的运行时错误与调试信息。

---


## 📕更新日志

- v0.1.0：推出 余波~ 的测试版。