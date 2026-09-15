# 部署

## 本机跑起来

```powershell
.\scripts\setup.ps1          # 建 .venv 并装依赖
Copy-Item .env.example .env  # 然后编辑 .env
.\scripts\run.ps1            # 起 uvicorn（默认 127.0.0.1:8000）
```

服务只监听回环地址，本机之外访问不到。

## 对外暴露：cloudflared 命名隧道

本机已装 `cloudflared`（`C:\Program Files (x86)\cloudflared\cloudflared.exe`）。
隧道**已经创建并绑定好**，不需要再 `cloudflared tunnel login`：

| 项 | 值 |
|---|---|
| 隧道名 | `skintone-api` |
| 隧道 ID | `c47b2860-c03e-4467-9bdf-006b1fe4d0d0` |
| 配置来源 | `config_src = cloudflare`（**远程托管**，ingress 在 Cloudflare 侧） |
| ingress | `skintest.0721.luxe` → `http://localhost:8000`，兜底 `http_status:404` |
| DNS | CNAME `skintest.0721.luxe` → `c47b2860-….cfargotunnel.com`，proxied |

因为配置是远程托管的，本地**不需要** `config.yml`，也**不需要** `cert.pem`。
连接器 token 存在仓库上一级的 `.cloudflared-token`（已在 `.gitignore` 中排除）。

```powershell
.\scripts\tunnel.ps1 run      # 用 TUNNEL_TOKEN 起连接器（token 不进命令行）
.\scripts\tunnel.ps1 status   # 查连接状态
.\scripts\tunnel.ps1 url      # 打印对外地址
```

隧道是**只出站**的：本机不开放任何入站端口，也不需要公网 IP。

## 暴露公网前的检查清单

逐条确认，任何一条不过就别开隧道：

- [ ] `SKINTONE_QUOTA_ENABLED=true`，且 `SKINTONE_QUOTA_PER_DAY_CLIENT` / `..._IP` 符合预期。
      这是默认的防滥用手段（5 次/天/指纹 + 5 次/天/IP）。**手机运营商 CGNAT 会让大量用户
      共用出口 IP**，真出现误伤就把 IP 上限抬高（例如 30），客户端保持 5
- [ ] 想真正上锁（而不只是限流）时才设 `SKINTONE_API_KEY`；默认配置下它是空的
- [ ] `SKINTONE_MAX_UPLOAD_MB` 已按磁盘余量确认（默认 20）
- [ ] `SKINTONE_RATE_LIMIT` 已确认（默认 30/minute）
- [ ] `SKINTONE_ALLOWED_ORIGINS` 只列前端实际域名，不要留 `*`
- [ ] `SKINTONE_RETENTION_DAYS` 与磁盘容量匹配；确认启动时清理逻辑生效
- [ ] 确认 `SKINTONE_HOST=127.0.0.1`，没有图省事写成 `0.0.0.0`
- [ ] 确认 `.env` 与 `.cloudflared-token` 都没被提交（`git status` 干净）
- [ ] 用 `curl.exe https://<域名>/v1/health` 确认经过隧道能通
- [ ] 故意连发超过额度的分析请求，确认第 6 次返回 `429`，且响应体 `details.remaining` 为 0
- [ ] 确认响应头里有 `X-Quota-Remaining`（注意经 Cloudflare 后头名会被小写化，
      浏览器的 `Headers.get()` 大小写不敏感，用 curl 手测时要加 `(?i)`）

## 为什么必须 HTTPS

前端跑在 GitHub Pages 上是 HTTPS。浏览器不会允许 HTTPS 页面去请求 HTTP 接口
（混合内容拦截），所以后端对外也必须是 HTTPS。cloudflared 隧道天然提供 HTTPS，
这也是选它而不是自己反代的原因。

## 前端地址配置

前端需要知道后端地址与访问密钥，都在页面里填、存 `localStorage`。
也可以在 URL 里用 `?api=https://…` 临时指定。
