# 夜班报告 2026-07-19(用户睡眠 6h 期间的自主推进)

## 完成的实验

### 1. 注入形式研究(logit-bias 软注入,120 runs)— 新结论

| 注入形式(el n=40) | hard-Δ | easy-Δ | 判定 |
|---|---|---|---|
| prompt 注入(无门控) | +14.1 | −18.6 | 强力但扰动 |
| prompt + oracle 门 | +13.3 | 0.0 | 强力+良性,需预测器 |
| uniform logit-bias +2/+4/+8 | +0.0/+1.3/+1.3 | ≈0 | **良性但无力**(+8 饱和) |

软注入假设一半成立:prompt 扰动伤害确实消失(证实 LCP 敏感性诊断),但
uniform bias 推不出模型不知道的术语(greedy 下翻不动,多 token 术语无接续
约束)。**指向 trie/前缀约束 contextual biasing**(ASR shallow-fusion 标准
做法:仅当已生成前缀匹配术语前缀时 boost 下一 token)= 可能同时强力+良性+
免门控的形式,需自定义 logits processor,列入完整系统。方法章已更新。

### 2. S2 数据层完备(Chinese-LiPS 全量 21 视频)

- 修复了 timeline JSON 抽取 bug(字段嵌套在 `paragraphs` 下)
- 全部 21 test 视频原时间轴重建:**663 分钟(11.1h),21/21 drift <10ms,0 缺失**
- 已上传 HF `chinese-lips-longform-debug/orig_timeline/`(1.2G)
- S2 现状:11.1h 原时间轴长语音 + 3,908 条幻灯片上下文机器草稿参考,全在 HF

### 3. ACL 60/60(S3)集成侦察 — 零障碍确认

- **视频由 ACL Anthology 官方托管**:`aclanthology.org/<id>.mp4` 规范直链
  (验证过 2022.acl-long.268)——三层里最干净的媒体恢复,无 ToS 灰区无 attrition
- 你的 `rasst-demo-acl6060-zh-segments`(5 talks/468 段/51.8min + SimulEval
  列表)可直接引导 split;方案+步骤在 docs/ACL6060_INTEGRATION.md,预估 1 天

### 4. 论文写作

- **06_experiments 成稿**:probe 协议、need-predictor 设置、full-system 计划
- method 章补注入形式研究结论

## 关键数字提醒(截至今晨的完整证据链)

premise:oracle +7.0 pooled (p=.03) / +18.8 hard (p<1e-4) →
抽取:VLM +14.1 hard(75% oracle)→ 门控:oracle-gate +7.3 pooled (p<.001) →
预测器:logprob AUC 0.62,hard 全保,easy 精度不足 →
注入形式:uniform bias 良性无力 → **完整系统方向:trie-constrained biasing**

## 下一步建议(按优先级)

1. ACL6060 集成执行(方案已写好,~1 天)
2. trie-constrained biasing logits processor 原型(完整系统核心件)
3. zh 侧 gating/bias 条件补全(16 段,完成双语对照)
4. 05/06/07 章的 full-system 留位填充随实验推进

## 注意事项

- hyper01 两个服务仍在运行(32B@8901, VL@8902),GPU 0/1 被占用;不用时
  `pkill -f "vllm serve"` 释放
- 本地 data_prep/ 现约 7G(chinese_lips 3.3G + mtedx_videos 3.2G + probe);
  HF 已验证后可清理
- 所有 runs/labels/timelines 归档在 docs/killtest/;提交历史 f0fe505→eb3ba7b
