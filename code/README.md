# Probabilistic Service-Region Routing

这是“具身清洁机器人概率服务区域路由与可证界联合规划”研究项目的第一版可复现实验骨架。

当前版本只解决论文成立前必须先验证的两个问题：

1. 能否用独立场景采样与单侧 Clopper-Pearson 置信下界，把目标附近的底盘候选筛成概率服务区域；
2. 在有限候选集上，能否用可采纳边下界和按需精算，在保持 `LB <= J* <= UB` 的同时减少昂贵边查询。

当前运动代价采用可控的非对称代理 oracle，用于验证算法逻辑和证书，不应作为论文实验结果。下一阶段需要把该 oracle 替换为 Hybrid A*、OMPL 或真实导航栈返回的可执行轨迹代价。

## 目录

- `docs/literature_gap_matrix.md`：10 篇本地论文的证据矩阵与创新边界。
- `docs/theory_notes.md`：与当前实现一致的命题、证书、风险边界和禁止表述。
- `docs/experiment_protocol.md`：尺度实验前冻结的指标、因子网格、排除规则和声称边界。
- `docs/pilot_report.md`：代理、状态格、Hybrid A* 和风险扫描的阶段证据。
- `src/psr_planner/risk.py`：概率服务区域候选生成、蒙特卡洛校准与精确二项置信下界。
- `src/psr_planner/routing.py`：分组路径动态规划、完整代价矩阵基线和带界惰性精算。
- `src/psr_planner/oracles.py`：非对称代价代理、离散 SE(2) 状态格 A* 及可采纳欧氏下界。
- `experiments/pilot.py`：端到端小规模预实验。
- `experiments/scaling_sweep.py`：冻结协议下的多规模、多候选和多障碍密度扫描。
- `experiments/screening_sweep.py`：冻结协议后的有向分组安全筛边探索性对照。
- `tests/`：风险校准与最优性证书单元测试。

## 运行

在本目录执行：

```powershell
$env:PYTHONPATH='src'
python -m unittest discover -s tests -v
python experiments/pilot.py --output results/pilot.json
python experiments/lattice_pilot.py --output results/lattice_pilot.json
python experiments/hybrid_pilot.py --output results/hybrid_pilot.json
python experiments/risk_sweep.py --output results/risk_sweep.json
python experiments/scaling_sweep.py --oracle proxy --targets 4,5,6 --candidates 2,3,4 --seeds 5 --output results/scaling_proxy.json
python experiments/scaling_sweep.py --oracle lattice --targets 4,5 --candidates 2,3 --seeds 2 --output results/scaling_lattice.json
python experiments/screening_sweep.py --oracle proxy --targets 4,5 --candidates 2,3 --seeds 2 --output results/screening_proxy.json
```

## 当前研究判据

只有同时满足以下条件，才继续按 CCF/较强 SCI 路线推进：

- 概率服务区域的置信下界能正确控制留出测试集上的失败率；
- 零间隙模式与完整精算基线得到相同最优值；
- 惰性算法显著减少精确边查询，并始终报告有效的上下界；
- 优势在目标数、候选数、障碍密度和不确定性变化下保持稳定。

若只能改善工程运行时间但无法给出有效风险校准或最优性证书，项目应降级为机器人应用型论文。
