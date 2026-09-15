> 语气与措辞待定稿（用户负责行文）。以下只写技术事实，不含宣传语。

# skintone-serve

肤色测量后端。**它不是通用 API 服务**：输入一张照片（可选带参考色卡），输出的是
CIELAB 三轴、ITA°、底色冷暖判定、置信度门禁，以及由此推导的色号与配色建议。

前端的采集页在另一个仓库。

## 它做什么

| 量 | 方法 |
|---|---|
| 深浅 | `ITA°`（Individual Typology Angle，皮肤科标准） |
| 底色冷暖 | CIELAB 色相角 `h_ab`、`a*/b*`、彩度 `C*`（彩度用于识别橄榄调） |
| 表面状态 | 高光占比、截断占比，独立于上面两轴 |

三条测量路线：

- **A 有卡**：`cv2.aruco` 定位参考卡 → 单应矫正 → 最小二乘 CCM → 逐块 ΔE00 自检
- **B 无卡**：语义遮罩后从巩膜/背景估光源 → McCamy 求 CCT → 投影到普朗克/日光轨迹做一维校正
- **C 交叉验证**：对数色度分解出黑色素/血红素指数，与 A/B 投票

**已知边界**（不要期待更多）：无卡模式下深浅可参考，冷暖与专家判定的一致率约 75–85%；
无参考物时做不到绝对精确。因此服务在证据不足时会返回 `confidence.level = "insufficient"`，
并把 `advice` 置为 `null`，而不是给一个看起来精确的数字。

算法细节与公式见 `docs/ALGORITHM.md`。

## 快速开始

```powershell
# 1. 建虚拟环境并装依赖
.\scripts\setup.ps1

# 2. 配置（至少设一个 API key）
Copy-Item .env.example .env
#   编辑 .env，填 SKINTONE_API_KEY

# 3. 起服务
.\scripts\run.ps1
```

## 验证

```powershell
# 单元测试（含 CIEDE2000 的 Sharma 34 组向量、合成图端到端自证）
.\.venv\Scripts\python.exe -m pytest tests -q

# 健康检查
curl.exe http://127.0.0.1:8000/v1/health
```

合成图自证测试的做法：程序渲染一张符合规格的参考卡 + 若干已知 CIELAB 真值的肤色块，
再施加已知色温的光源偏移，走完整 `POST /v1/analyze`，断言还原出的 `labD65` 与真值
`ΔE00 < 3`。见 `tests/test_synthetic_acceptance.py`。

## 数据与留存

- 元数据存 SQLite，原图归档到 `SKINTONE_DATA_DIR/archive/{yyyy-mm}/`
- 落盘前剥离 EXIF（保留方向与时间戳）
- 启动时清理超过 `SKINTONE_RETENTION_DAYS`（默认 30）的归档
- `DELETE /v1/result/{id}` 会同时删除元数据与归档文件
- `meta.consent.storeImage = false` 时只在本进程内存中处理，**不落盘原图**

## 对外暴露

服务默认只监听 `127.0.0.1`。对外通过 cloudflared 命名隧道（只出站，不开放入站端口）：

```powershell
.\scripts\tunnel.ps1 run      # 起隧道
.\scripts\tunnel.ps1 status   # 查连接状态
.\scripts\tunnel.ps1 url      # 打印对外地址
```

暴露公网前的检查清单见 `docs/DEPLOY.md`——**尤其别把 `SKINTONE_API_KEY` 留空**。

## 目录结构

```
skintone/
  api.py         路由
  config.py      全部阈值与常量（不要散落到函数体里）
  schemas.py     请求/响应模型
  storage.py     SQLite + 原图归档
  analysis.py    三条路线的编排
  color/         srgb / ciede2000 / adaptation / illuminant / skin / palette
  vision/        card（ArUco + CCM）/ roi（掩膜与鲁棒统计）/ face（Haar 兜底）
  cards/         参考卡几何与色块值
scripts/         setup / run / tunnel
tests/           单测与合成图自证
docs/            ALGORITHM / API / DEPLOY
```

## 接口

见 `docs/API.md`。字段级契约（含参考卡几何与色块数值）见仓库根目录上一级的 `CONTRACT.md`。

## 许可

MIT
