# Linux 关卡任务 --- 完成SSH连接与端口映射并运行`hello_world.py`

## 1. 登录开发机
用Internstudio IO登录开发机：

## 2. 创建项目
选择 微小规模训练 等级资源即可，命名开发机为`Linux`:

点击立即创建后，等待开发机状态从`排队中`变为`运行中`便可使用:

点开SSH信息并复制:


## 3. VScode SSH登录
首先在VScode上下载 Remote - SSH 扩展:

然后再电脑屏幕图标里面点击齿轮按钮配置SSH:

将复制的SSH登录信息复制到输入栏内，并摁回车两次:

此时VScode右下角会出现Connect按钮，点击后弹出新VScode窗口，输入之前开发机SSH登录密码即可远程登录:


## 4. 创建端口映射
点击SSH加号旁边的齿轮按钮，添加`LocalForward 7860 127.0.0.1:7860`到配置文件中:


在开发机上创建`hello_world.py`文件并输入以下内容:
```python
import socket
import re
import gradio as gr
 
# 获取主机名
def get_hostname():
    hostname = socket.gethostname()
    match = re.search(r'-(\d+)$', hostname)
    name = match.group(1)
    
    return name
 
# 创建 Gradio 界面
with gr.Blocks(gr.themes.Soft()) as demo:
    html_code = f"""
            <p align="center">
            <a href="https://intern-ai.org.cn/home">
                <img src="https://intern-ai.org.cn/assets/headerLogo-4ea34f23.svg" alt="Logo" width="20%" style="border-radius: 5px;">
            </a>
            </p>
            <h1 style="text-align: center;">☁️ Welcome {get_hostname()} user, welcome to the ShuSheng LLM Practical Camp Course!</h1>
            <h2 style="text-align: center;">😀 Let’s go on a journey through ShuSheng Island together.</h2>
            <p align="center">
                <a href="https://github.com/InternLM/Tutorial/blob/camp3">
                    <img src="https://oss.lingkongstudy.com.cn/blog/202406301604074.jpg" alt="Logo" width="20%" style="border-radius: 5px;">
                </a>
            </p>

            """
    gr.Markdown(html_code)

demo.launch()
```

然后安装依赖:
```bash
pip install gradio==4.29.0
```
通过执行`python hellow_world.py`运行:

进入网站:

