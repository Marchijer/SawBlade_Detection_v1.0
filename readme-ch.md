# 缺陷检测和分割项目

## 目录
- [概述](#概述)
- [安装](#安装)
- [使用](#使用)
- [功能](#功能)
- [配置](#配置)
- [贡献](#贡献)
- [许可证](#许可证)
- [联系](#联系)

## 概述

​	该项目使用 YOLO 模型对图像进行实例分割，然后进行缺陷检测。其包含一个基于 PyQt5 的 GUI 应用程序，用于监控特定目录中的新图像，使用模型处理这些图像，并在 GUI 中显示结果。

## 安装

### 前提条件
- Python 3.x
- 所需的 Python 库（在 `requirements.txt` 中列出）

### 步骤
1. 进入项目目录：

```sh
cd path_to_your_project
```

2. 安装依赖项：

```sh
pip install -r requirements.txt
```

## 使用

### 运行项目
要运行项目，请使用以下命令：
```sh
python main.py
```

### 示例
1. 将要处理的图像放入 `main.py` 中指定的目录中，具体操作为将摄像头控制软件（MSV）中三个摄像机拍摄的**图像保存路径**分别赋值给`main.py`文件中的三个变量： `folder_to_watch`, `folder_to_watch2`, `folder_to_watch3`
2. 运行脚本：
   ```sh
   python main.py
   ```
3. GUI 将显示最新处理的图像及检测结果。

## 功能
- 监控目录中出现的新图像
- 使用 YOLO 进行实例分割
- 使用 YOLO 进行缺陷检测
- 基于 PyQt5 的 GUI 显示结果

## 配置
确保在脚本中正确设置以下配置：

- 模型路径：
  ```python
  box_selection_model = YOLO("pt/segmentation/best.pt")  # 分割模型
  detect_model = YOLO("pt/detection/best.pt")  # 检测模型
  ```
- 输入目录：
  ```python
  self.folder_to_watch = r".\images1"
  self.folder_to_watch2 = r".\images2"
  self.folder_to_watch3 = r".\images3"
  ```

## 贡献
1. Fork 此仓库。
2. 创建一个新分支 (`git checkout -b feature-branch`)。
3. 提交你的更改 (`git commit -m 'Add some feature'`)。
4. 推送到分支 (`git push origin feature-branch`)。
5. 打开一个 Pull Request。

## 许可证
此项目依据 MIT 许可证发布 - 详见 LICENSE 文件。

## 联系
维护者联系方式 - [your.email@example.com](mailto:your.email@example.com)
GitHub: [https://github.com/yourusername](https://github.com/yourusername)

