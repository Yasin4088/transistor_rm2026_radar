# 北航Transistor战队27赛季雷达站代码

## 项目运行

1.编译Mid-70激光雷达的SDK接口代码

```bash
cd ./tools/livox_sdk
colcon build
```

2.构建conda虚拟环境（实机python版本为3.12可跳过此步）

```bash
cd ~/transistor_rm2026_radar
conda create -n yolov12_forRadar python=3.12
conda activate yolov12_forRadar
```

3.安装项目运行所需依赖

```bash
pip freeze >requirements.txt
pip install -r requirements.txt
```

4.运行主函数

```bash
python ./src/main.py
```