# Security

## Scope

SparkDuet is a serving recipe and tooling. It runs inference servers on your own
hardware and exposes OpenAI-compatible HTTP endpoints.

## Reporting

Please open a GitHub issue labeled `security`. Do not include cluster secrets,
tokens, or private network details in the issue body.

## Operator guidance (read before first start)

- The API binds `0.0.0.0` by default in the lane compose files so the router can
  reach backends on both nodes. If you do not need LAN access, bind backends to
  loopback and reach them over SSH tunnels, or put your own authenticated
  gateway in front. **SparkDuet ships no authentication of its own.**
- Never commit `sparkduet.env`, it contains your cluster's addresses. It is
  gitignored; keep it that way.
- Weights are pinned by revision and verified at prepare time; keep
  `HF_HUB_OFFLINE=1` after caches are warm.
- Containers run with host networking and IPC as required by vLLM multi-node
  serving; treat the nodes as trusted single-tenant machines.
- Third-party images, models, and kernels carry their own licenses and security
  postures (see `CREDITS.md`).
