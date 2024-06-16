# Defect Detection and Segmentation

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Description

This project utilizes YOLO models to perform defect detection and segmentation on images. It includes a PyQt5-based GUI application that monitors specific directories for new images, processes them using the models, and displays the results in the GUI.

## Installation

### Prerequisites
- Python 3.x
- Required Python libraries (listed in `requirements.txt`)

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/yourproject.git

2. Navigate to the project directory:
   ```sh
   cd yourproject
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Project
To run the project, use the following command:
```sh
python main.py
```

### Example
1. Place the images you want to process in the directories specified in `main.py` (`folder_to_watch`, `folder_to_watch2`, `folder_to_watch3`).
2. Run the script:
   ```sh
   python main.py
   ```
3. The GUI will display the latest processed images and the detection results.

## Features
- Defect detection using YOLO
- Image segmentation using YOLO
- Directory monitoring for new images
- PyQt5-based GUI for displaying results

## Configuration
Ensure the following configurations are set correctly in the script:

- Model paths:
  ```python
  box_selection_model = YOLO("pt/segmentation/best.pt")  # segment
  detect_model = YOLO("pt/detection/best.pt")  # detect
  ```
- Input directories:
  ```python
  self.folder_to_watch = r".\images1"
  self.folder_to_watch2 = r".\images2"
  self.folder_to_watch3 = r".\images3"
  ```

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Maintainer Name - [your.email@example.com](mailto:your.email@example.com)
GitHub: [https://github.com/yourusername](https://github.com/yourusername)
```

将上述内容保存为`requirements.txt`和`README.md`文件。这些文件提供了安装项目所需的依赖项和如何使用项目的详细说明。确保根据你的实际项目需求，调整和补充内容，例如更新GitHub仓库地址、你的联系信息及其他必要说明。