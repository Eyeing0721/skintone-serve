# HTTP 接口

基地址默认 `http://127.0.0.1:8000`。字段级完整契约（含参考卡几何、色块数值、
置信度门禁阈值、错误码表）见仓库上一级的 `CONTRACT.md`，此处只列接口。

**认证**：设置了 `SKINTONE_API_KEY` 时，除 `GET /v1/health` 外所有请求都要带
`X-API-Key` 头；不匹配返回 `401`。

**限流**：按 IP 令牌桶，默认 `SKINTONE_RATE_LIMIT`（30/minute），超限返回 `429`。

---

## GET /v1/health

免认证。用于判断前端与后端是否匹配。

```json
{ "status": "ok", "version": "0.1.0", "specVersion": "1.0.0",
  "uptimeSeconds": 12.3, "cards": ["skintone-a4-v1"], "authRequired": true }
```

`authRequired` 直接反映 `SKINTONE_API_KEY` 是否非空——前端据此决定要不要提示用户填密钥。

## GET /v1/card/{card_id}

返回可打印参考卡的完整规格，供前端渲染，避免两个仓库的色块值漂移。
未知 id 返回 `404`。当前只有 `skintone-a4-v1`。

## POST /v1/analyze

`multipart/form-data`：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `image` | file | 是 | JPEG/PNG，上限 `SKINTONE_MAX_UPLOAD_MB` |
| `meta` | text(JSON) | 是 | `mode` / `capture` / `consent` / `clientVersion` 等 |
| `rois` | text(JSON) | 否 | 原图像素坐标的闭合多边形；缺省时服务端用 Haar 兜底自动检测 |

`meta.mode` 取 `nocard` | `card` | `calibrate`。

响应包含 `confidence`（含逐条门禁的实测值与阈值）、`illuminant`、`calibration`、
`skin`（`labD65` / `itaDeg` / `depthClass` / `hueAngleDeg` / `chroma` / `undertone` /
`melaninIndex` / `hemoglobinIndex`）、`advice`、`warnings`、`disclaimer`。

`confidence.level = "insufficient"` 时 `advice` 为 `null`——这是有意设计，
不是缺陷。请前端据此引导用户改做物理试色。

## POST /v1/card/{card_id}/calibrate

色卡自标定。用户在自己日光下拍一张卡，反推这张**纸**实际印出来的颜色，
结果按 `profileId` 存档，之后分析时用实测值代替标称值。

响应含 `measuredSrgb`、`grayNeutrality`（灰阶条是否够中性）、`quality`。

> 注意：标定只恢复**相对**颜色，整体光照增益不可知；中性轴由灰阶条锁定。

## GET /v1/result/{request_id}

取回已存档的去标识化结果。不回传原图。

## DELETE /v1/result/{request_id}

删除该记录的元数据与归档原图。成功返回 `204`。

## GET /v1/stats

`{ "totalRequests": 0, "storedImages": 0, "diskBytes": 0 }`。不含任何用户数据。

---

## 错误格式

所有错误统一为：

```json
{ "error": { "code": "CARD_NOT_DETECTED", "message": "…", "hint": "…", "details": {} } }
```

错误码：`BAD_IMAGE` `CARD_NOT_DETECTED` `CARD_PROFILE_NOT_FOUND` `ROI_TOO_SMALL`
`FACE_NOT_FOUND` `ILLUMINANT_UNRELIABLE` `PAYLOAD_TOO_LARGE` `UNAUTHORIZED`
`RATE_LIMITED` `INTERNAL`。
