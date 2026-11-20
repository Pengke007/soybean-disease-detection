# 🌱 SoyGuard 大豆病害智能识别

一个基于深度学习的大豆叶片病害识别系统，支持**网页端**与**微信小程序**两种形态。
用户拍摄/上传大豆叶片照片，系统返回病害类别、置信度、症状特征、易混淆区分与科学防治建议。

---

## 一、项目结构

```
soybean-app/
├── backend/                 # 后端服务与训练脚本
│   ├── app.py               # Flask API 服务（图片识别 / 知识库 / 托管网页）
│   ├── diseases.py          # 病害知识库（症状、病原、防治建议）
│   ├── prepare_data.py      # 将 parquet 数据集导出为图片并按类别划分
│   └── train.py             # MobileNetV3 迁移学习训练脚本
├── frontend/
│   └── index.html           # 网页前端（拍照/上传/拖拽/粘贴识别）
├── miniprogram/             # 微信小程序工程（可直接用微信开发者工具打开）
│   ├── app.js / app.json / app.wxss
│   ├── pages/index/         # 识别页
│   ├── pages/library/       # 病害图鉴
│   ├── pages/detail/        # 病害详情
│   └── assets/              # tabBar 图标
├── data/
│   ├── raw/                 # 原始数据集（parquet）
│   └── processed/           # 处理后图片（train/val/test/类别）
├── models/                  # 训练产出：权重、类别索引、训练报告
└── docs/                    # 补充说明文档
```

---

## 二、数据集来源

| 数据集 | 类别 | 数量 | 许可 | 用途 |
|---|---|---|---|---|
| `anandvermagmailcom/soybean-leaf-diseases`（Hugging Face） | 细菌性角斑病、蛙眼叶斑病、健康、大豆锈病 | 1800 张（train 1260 / val 268 / test 272） | CC BY 4.0 | **模型训练与评估** |
| 病害知识库（本项目整理） | 另含褐斑病、尾孢叶斑病、猝死综合征、花叶病毒病、白粉病、靶斑病、霜霉病 | — | 科普用途 | **图鉴展示**（当前模型未训练这些类别） |

> 当前模型可识别 4 个类别；知识库覆盖 11 类常见大豆病害供科普参考。
> 如需扩展识别类别，按 `prepare_data.py` 增加数据后重新训练即可（见第六节）。

---

## 三、快速开始

### 1. 环境准备

```bash
# 使用项目隔离环境（已创建于 .workbuddy/binaries/python/envs/soybean）
# 依赖：torch torchvision flask flask-cors pandas pyarrow pillow numpy scikit-learn
```

### 2. 数据准备（首次运行）

```bash
cd soybean-app/backend
python prepare_data.py
```

### 3. 训练模型

```bash
python train.py
```

训练完成后会在 `models/` 生成：
- `soybean_model.pth` — 模型权重
- `class_indices.json` — 类别索引
- `training_report.json` — 训练报告（准确率、F1、混淆矩阵、每轮曲线）

### 4. 启动服务

```bash
python app.py
```

浏览器打开 <http://localhost:5000> 即可使用网页版。
若未训练模型，服务会以**演示模式**运行（界面可联调，但结果非真实诊断）。

### ✅ 当前模型实测指标（已完成训练）

| 指标 | 数值 |
|---|---|
| 模型 | MobileNetV3-Small（ImageNet 预训练迁移学习） |
| 测试集准确率 | **99.26%** |
| 测试集加权 F1 | **0.9926** |
| 最佳验证集准确率 | 98.13% |
| 训练/验证/测试样本 | 1260 / 268 / 272 |
| 类别 | 细菌性角斑病、蛙眼叶斑病、健康、大豆锈病 |

各类别测试集表现：

| 类别 | 精确率 | 召回率 | F1 |
|---|---|---|---|
| 细菌性角斑病 | 0.971 | 1.000 | 0.986 |
| 蛙眼叶斑病 | 1.000 | 0.971 | 0.985 |
| 健康叶片 | 1.000 | 1.000 | 1.000 |
| 大豆锈病 | 1.000 | 1.000 | 1.000 |

完整训练曲线与混淆矩阵见 `models/training_report.json`。

---

## 四、微信小程序接入

1. 用微信开发者工具「导入项目」，目录选择 `soybean-app/miniprogram`，填入你的 AppID（可用测试号）。
2. 修改 `miniprogram/app.js` 中的 `apiBase` 为后端服务器地址：
   - 本地调试：`http://localhost:5000`，并在开发者工具「详情 → 本地设置」勾选 **不校验合法域名**。
   - 真机/上线：必须是 **HTTPS 域名**，并在微信公众平台「开发管理 → 服务器域名」中配置为 request 合法域名。
3. 后端需要公网可访问（部署到云服务器，或内网穿透用于调试）。
4. 小程序端通过 `wx.uploadFile` 上传图片到 `/api/predict`，后端已兼容 multipart 上传。

> 微信小程序要求正式版后端域名必须备案且支持 HTTPS，个人开发者建议使用云开发或已有备案域名。

---

## 五、API 接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/health` | GET | 健康检查、模型是否加载、类别列表 |
| `/api/diseases` | GET | 病害知识库 |
| `/api/predict` | POST | 识别。支持 `multipart/form-data`（字段名 `image`）或 JSON `{"image": "data:image/...;base64,...`} |
| `/api/predict/base64` | POST | 识别（仅 JSON base64） |

返回示例：
```json
{
  "mode": "model",
  "top": {"class": "Soyabean_rust", "name_zh": "大豆锈病", "confidence": 96.3, "is_disease": true},
  "candidates": [...],
  "detail": {"name_zh": "大豆锈病", "pathogen": "...", "symptoms": "...", "control": [...]}
}
```

---

## 六、如何扩展识别更多病害

1. 收集目标病害图片，按 `data/processed/{train,val,test}/{类别名}/` 目录结构放置。
2. 更新 `data/classes.json` 类别列表。
3. 在 `backend/diseases.py` 补充对应类别的知识条目。
4. 重新运行 `python train.py`。

---

## 七、免责声明

本系统识别结果**仅供参考**，不能替代专业植保诊断。实际防治请结合田间症状并遵循当地农技部门指导。
数据集与图片版权归原作者所有，仅用于学习研究。
