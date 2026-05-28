$xmlPath = 'D:\Mytest\tmp_mid_report_docx\unzipped\word\document.xml'
[xml]$doc = Get-Content -LiteralPath $xmlPath -Encoding UTF8
$ns = New-Object System.Xml.XmlNamespaceManager($doc.NameTable)
$ns.AddNamespace('w','http://schemas.openxmlformats.org/wordprocessingml/2006/main')
$body = $doc.SelectSingleNode('//w:body', $ns)
$paras = $doc.SelectNodes('//w:body/w:p', $ns)
$wNs = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

function Set-ParagraphText {
    param([int]$Index, [string]$Text)
    $p = $paras[$Index]
    $runs = @($p.SelectNodes('./w:r', $ns))
    foreach ($r in $runs) { [void]$p.RemoveChild($r) }
    $rNew = $doc.CreateElement('w','r',$wNs)
    $tNew = $doc.CreateElement('w','t',$wNs)
    $xmlSpace = $doc.CreateAttribute('xml','space','http://www.w3.org/XML/1998/namespace')
    $xmlSpace.Value = 'preserve'
    [void]$tNew.Attributes.Append($xmlSpace)
    $tNew.InnerText = $Text
    [void]$rNew.AppendChild($tNew)
    [void]$p.AppendChild($rNew)
}

function New-ParagraphLike {
    param([System.Xml.XmlNode]$Like, [string]$Text)
    $newP = $Like.CloneNode($true)
    $runs = @($newP.SelectNodes('./w:r', $ns))
    foreach ($r in $runs) { [void]$newP.RemoveChild($r) }
    $rNew = $doc.CreateElement('w','r',$wNs)
    $tNew = $doc.CreateElement('w','t',$wNs)
    $xmlSpace = $doc.CreateAttribute('xml','space','http://www.w3.org/XML/1998/namespace')
    $xmlSpace.Value = 'preserve'
    [void]$tNew.Attributes.Append($xmlSpace)
    $tNew.InnerText = $Text
    [void]$rNew.AppendChild($tNew)
    [void]$newP.AppendChild($rNew)
    return $newP
}

function Insert-ParagraphsAfter {
    param([int]$Index, [string[]]$Texts)
    $ref = $paras[$Index]
    foreach ($text in $Texts) {
        $newP = New-ParagraphLike -Like $ref -Text $text
        [void]$body.InsertAfter($newP, $ref)
        $ref = $newP
    }
}

Set-ParagraphText 16 '论文总体目标是在算力互联网架构下，设计一种融合强化学习、图注意力网络和上下文感知动态奖励机制的多机器人算力调度方法。当前项目已从初始小规模环境扩展到20个机器人、10个异构算力节点的主实验配置，智能体能够根据任务类型、任务优先级、任务规模、deadline、机器人电量、机器人队列长度、节点空闲算力、节点负载、节点时延、拓扑距离以及通信风险等状态信息，自主选择合适的本地、边缘、区域或云端计算节点。通过多seed训练和仿真实验，验证PPO-GAT-Scoring相对于普通PPO、朴素PPO+GAT以及随机、轮询、贪心等基准策略在平均奖励、总完成时间、队列等待、能耗、deadline违约和节点过载等指标上的改进。'
Set-ParagraphText 17 '具体目标包括：第一，建立面向20机器人、10节点异构算力互联网场景的仿真环境，支持多机器人异构画像、异构计算节点、任务类型差异、拓扑传输代价、邻接拥塞传播、机器人低电量和充电恢复过程；第二，完成PPO-Baseline、PPO-GAT-Naive和节点打分式PPO-GAT-Scoring三类策略的设计与实现；第三，构造面向多场景上下文感知的动态奖励函数，能够根据高负载、低电量、紧急任务、高通信时延以及正常场景自动调整能耗、deadline、队列、过载、延时和负载均衡等权重；第四，形成四类单独场景和混合多场景训练验证流程，并输出多seed评估CSV、per-seed明细和可视化图表，为论文实验章节提供数据支撑。'
Set-ParagraphText 21 '第三是实验验证层，当前实验已从传统基准算法对比扩展为PPO模型族对比和多场景动态验证。传统基准包括Random-Policy、RoundRobin-Policy和GreedyCPU-Policy；学习模型包括PPO-Baseline、PPO-GAT-Naive和PPO-GAT-Scoring。实验指标包括平均奖励、总奖励、平均计算时间、平均排队延迟、平均网络时延、平均总完成时间、平均能耗、deadline惩罚、过载惩罚、上下文场景得分以及机器人充电次数等。同时设计低电量、高负载、紧急任务、高延迟/低带宽四类专门场景，以及一次训练中同时包含正常与特殊场景的mixed-context配置，用于检验动态奖励机制与节点打分式GAT结构在复杂动态条件下的有效性。'
Set-ParagraphText 23 '预期成果包括：一是形成完整的多机器人算力互联网调度仿真代码，包括环境建模、动态场景识别、上下文感知奖励函数、算法训练、模型评估、基线对比、结果绘图和轻量系统验证脚本；二是形成可复现实验数据，包括训练模型、TensorBoard日志、多seed评估CSV、per-seed明细、阶段性图表和系统验证事件日志；三是完成论文主体方法，包括多要素协同建模、节点打分式PPO+GAT、动态多场景奖励函数和机器人低电量充电恢复机制；四是形成论文或技术报告性质的研究成果，系统阐述算力互联网背景下的多机器人智能调度方法与实验结论。'
Set-ParagraphText 26 '截至本次中期检查，项目已完成文献梳理、核心仿真环境搭建、20机器人/10节点规模扩展、三类强化学习/图强化学习策略实现、传统基线对比、PPO模型族多seed评估、四类专门场景实验、混合多场景动态奖励实验、机器人充电机制、图表生成和轻量系统验证。整体进度已超过开题计划中2026年4月至5月阶段的基本要求，并已形成可支撑论文方法章节、实验章节和创新点总结的阶段性数据。下一阶段重点将从模型有效性验证转向消融实验、统计显著性分析、图表完善和论文正文写作。'
Set-ParagraphText 32 '目前项目已经形成面向20个机器人、10个异构算力节点的多机器人智能体算力调度实验环境，核心文件包括configs/env_20r_10n.yaml、configs/scenarios/*.yaml、src/envs/multi_robot_scheduler_env.py、src/envs/task.py、src/envs/robot.py和src/envs/node.py。环境采用Gymnasium接口，动作空间为离散计算节点选择，观测空间包含当前任务特征、当前任务来源机器人状态、所有计算节点状态和所有机器人状态。当前主配置包含20个机器人和10个计算节点，节点类型覆盖local、edge、regional和cloud，CPU容量从4.0到28.0不等，基础时延从0.04到0.85不等，能耗因子从0.78到1.55不等，能够模拟本地、边缘、区域和云端算力资源的异构差异。'
Set-ParagraphText 33 '机器人侧已经实现异构画像。不同机器人具有不同home_node_id、本地CPU、任务到达率、任务规模偏好和deadline偏好。任务侧支持navigation、perception、mapping、manipulation、inspection五类任务，不同任务类型具有不同的规模系数、deadline系数、优先级偏置、本地计算需求和传输需求。环境每一步根据机器人任务产生率随机生成任务，任务按优先级、deadline、来源机器人和任务编号排序后进入中央调度器决策流程，从而模拟多机器人连续产生任务、智能体逐个选择计算节点的动态过程。'
Set-ParagraphText 35 '2.3 上下文感知动态奖励函数与场景识别'
Set-ParagraphText 36 '在原有多目标奖励函数基础上，项目已进一步实现面向多场景上下文感知的动态奖励机制。当前奖励函数以任务总完成时间、能耗、deadline违约、节点过载、拓扑距离和机器人队列长度为基础惩罚项，同时加入系统backlog惩罚、节点压力惩罚、远程云节点惩罚、云端误用惩罚、本地性奖励、deadline余量奖励和云端拥塞缓解奖励。动态奖励机制会根据当前状态计算context_load_context、context_energy_risk、context_urgency_level、context_comm_risk、context_compute_pressure等上下文因子，并进一步推断inferred_scenario及low_energy、high_load、emergency、high_latency、normal等场景得分。'
Set-ParagraphText 37 '在低电量场景下，奖励函数会提高能耗和通信延迟相关权重，引导策略避免不必要的远端高能耗节点；在高负载场景下，提高队列、过载和负载均衡权重，引导策略避开拥塞节点；在紧急任务场景下，提高deadline、队列和slack相关权重；在高延迟/低带宽场景下，提高通信延迟和能耗权重。比较脚本和轻量验证脚本已统一输出avg_context_load、avg_context_energy_risk、avg_context_urgency、avg_context_comm_risk、avg_scenario_low_energy_score、avg_scenario_high_load_score、avg_scenario_emergency_score、avg_scenario_high_latency_score等字段，用于检验动态奖励是否按场景触发。'
Set-ParagraphText 38 '2.4 PPO基线与传统策略对比实验'
Set-ParagraphText 39 '已实现PPO-Baseline、Random-Policy、RoundRobin-Policy和GreedyCPU-Policy对比。在20机器人、10节点配置下，最终模型评估中RoundRobin-Policy平均奖励约为-2.0005，Random-Policy约为-2.1220，PPO-Baseline最终检查点约为-3.0320，GreedyCPU-Policy约为-5.7510；当采用PPO训练过程中的best_model进行评估时，PPO-Baseline平均奖励约为-1.4585，优于RoundRobin-Policy、Random-Policy和GreedyCPU-Policy。该结果说明PPO策略对模型保存时机较敏感，best_model能够更好反映训练过程中学到的有效策略。'
Set-ParagraphText 40 '从指标上看，PPO-Baseline best_model相对于传统基线在过载惩罚和能耗控制方面具有优势，但在部分场景下队列延迟仍高于RoundRobin等简单均衡策略，说明普通PPO虽然能够学习多目标奖励，但缺少对计算节点结构和候选动作语义的显式刻画。因此，后续PPO-GAT-Scoring通过节点级评分机制进一步增强对节点负载、时延、拓扑距离和任务需求之间关系的表达。'
Set-ParagraphText 41 '2.5 PPO+GAT融合策略实现与20机器人/10节点多seed评估'
Set-ParagraphText 43 '在20机器人、10节点主配置下，多seed评估表明PPO-GAT-Scoring整体最优。根据outputs/results/ppo_gat_comparison_20r_10n_best.csv，PPO-GAT-Scoring平均奖励约为0.5596，PPO-GAT-Naive约为-0.4988，PPO-Baseline约为-1.4585。相对于普通PPO和朴素GAT，节点打分式PPO-GAT-Scoring在平均奖励、队列延迟、总完成时间和过载控制方面表现更稳定，说明将图嵌入与候选计算节点动作选择对齐是提升多机器人算力调度性能的关键。'
Set-ParagraphText 44 '从分项指标看，PPO-GAT-Scoring能够更好识别拓扑距离、节点负载和任务deadline之间的权衡关系。PPO-GAT-Naive虽然引入了图结构，但缺少显式节点打分约束，容易出现排队延迟偏高和动作选择不稳定的问题；PPO-Baseline虽然具备一定学习能力，但对异构节点之间的结构关系利用不足。该结果为论文创新点提供了阶段性证据：图神经网络不应只作为普通状态编码器，而应服务于节点级决策和动作语义建模。'
Set-ParagraphText 45 '2.6 多场景动态奖励实验与混合场景验证'
Set-ParagraphText 46 '为验证动态奖励函数和场景识别机制，项目构造了low_energy_20r_10n、high_load_24r_10n、emergency_20r_10n和high_latency_20r_12n四类专门场景，并分别训练PPO-Baseline、PPO-GAT-Naive和PPO-GAT-Scoring。在四个场景中，PPO-GAT-Scoring均取得最高平均奖励，并在每个场景的5个评估seed中全部胜出。其中低电量场景Scoring平均奖励约为-1.893，优于PPO-Baseline的-3.661和PPO-GAT-Naive的-6.153；高负载场景Scoring约为-9.909，优于PPO-Baseline的-14.591和Naive的-15.507；紧急任务场景Scoring约为-12.035，优于Naive的-23.185和PPO的-25.857；高延迟/低带宽场景Scoring约为-3.283，优于Naive的-9.300和PPO的-9.724。'
Set-ParagraphText 47 '在进一步的mixed_context_20r_10n实验中，环境在一次训练过程中同时包含正常任务、低电量、充电恢复、高负载、紧急任务和高延迟/低带宽等动态过程。最新评估结果显示，PPO-GAT-Scoring平均奖励约为-10.810，PPO-Baseline约为-15.539，PPO-GAT-Naive约为-20.507；Scoring的平均总完成时间约为1.420，显著低于PPO-Baseline的1.742和Naive的2.077；平均队列延迟约为0.098，显著低于PPO-Baseline的0.384和Naive的0.733；deadline惩罚和过载惩罚也明显更低。混合场景中avg_context_energy_risk约为0.288，avg_scenario_low_energy_score约为0.307，avg_charging_robots约为0.034，total_charging_starts约为1.4，total_charging_recoveries约为1.2，说明低电量和充电过程已经真实进入训练与评估轨迹。'
Set-ParagraphText 48 '2.7 结果可视化与实验流程整理'
Set-ParagraphText 49 '项目已完成训练脚本、对比脚本、绘图脚本和一键实验脚本整理。scripts目录下保留了train_ppo_baseline.ps1、train_ppo_naive.ps1、train_ppo_scoring.ps1、compare_ppo_models.ps1、compare_baselines.ps1、plot_metric_panel.py、run_gat_experiment.ps1、run_ppo_model_scenario.ps1、run_all_ppo_model_scenarios.ps1和run_mixed_context_ppo_models.ps1等关键脚本。outputs/results目录保存了PPO/GAT对比、传统基线对比、四类单独场景对比和mixed-context对比CSV；outputs/figures目录保留metric_panel.png、metric_panel_best.png等核心图表。'
Set-ParagraphText 50 '当前图表主要围绕平均奖励、总完成时间、队列延迟、deadline惩罚、过载惩罚和能耗成本六类指标展开，能够直观展示PPO-GAT-Scoring相对于PPO-Baseline和PPO-GAT-Naive的优势。实验流程已经具备较强可复现性，后续将在固定最终训练配置后补充消融实验图、混合场景图和统计显著性说明。'
Set-ParagraphText 51 '2.8 轻量系统验证'
Set-ParagraphText 52 '为支撑《多机器人任务持续产生—中央调度器决策—计算节点执行》的系统级论证，项目已完成轻量系统验证脚本src/lightweight_multi_robot_validation.py。该验证不依赖Gazebo，在当前阶段用于快速检查算法调度闭环是否成立。验证脚本已经支持导出事件级日志和summary结果，包括任务来源机器人、所选节点、节点类型、总完成时间、队列延迟、网络时延、能耗、deadline惩罚、过载惩罚、动态奖励权重、自动识别场景得分以及机器人充电状态等。'
Set-ParagraphText 53 '轻量验证的意义在于，它将算法评估从单纯离线指标扩展到持续任务流过程，能够记录每一步任务生成、场景识别、奖励调权和节点选择结果，为论文后续描述原型系统和验证流程提供材料。尽管该验证尚未达到Gazebo/ROS2物理仿真的精细程度，但已经完成中央调度器、任务产生、节点执行、机器人电量变化和充电恢复的闭环，是下一阶段接入更真实机器人仿真环境的基础。'
Set-ParagraphText 54 '2.9 论文发表及专利申请情况'
Set-ParagraphText 64 '第四项关键技术是面向多场景上下文感知的动态奖励函数。与固定权重奖励不同，当前方法会根据任务类型、机器人电量、机器人队列、节点负载、任务紧急程度、通信风险和计算压力自动推断当前状态更接近低电量、高负载、紧急任务、高延迟或正常场景，并据此调整能耗、deadline、队列、延迟、过载和负载均衡等权重。该机制使智能体能够在低电量时更重视节能，在紧急任务时更重视deadline，在高负载时更重视过载和队列，在高延迟时更重视通信代价。'
Set-ParagraphText 65 '该奖励函数的难点在于平衡不同指标量纲和优化方向。如果时延惩罚过强，策略可能过度选择近端节点导致过载；如果云端惩罚过强，策略无法在边缘拥塞时利用云端资源；如果能耗惩罚过强，策略可能牺牲deadline。当前项目通过四类专门场景和mixed-context混合场景进行验证，结果显示PPO-GAT-Scoring在不同场景中均取得最高平均奖励，并能在机器人电量下降、进入充电和恢复任务生成的过程中保持较低的队列延迟和过载惩罚。'
Set-ParagraphText 67 '第五项关键技术是可复现实验流程。项目已经将环境配置、模型训练、PPO/GAT比较、传统基线比较、四类场景实验、mixed-context混合场景实验、绘图和轻量验证分别封装成脚本，结果统一输出到outputs目录。多seed评估表、per_seed明细、场景得分字段和充电状态字段为后续论文统计分析提供基础。这一工作虽然不属于算法创新本身，但对于学位论文非常关键，因为调度算法的有效性必须通过可重复、可追踪、可解释的实验数据来支撑。'
Set-ParagraphText 70 '当前工作仍存在若干不足。第一，虽然主实验已扩展到20个机器人和10个节点，并补充了24机器人/10节点与20机器人/12节点场景，但尚未系统验证更大规模下的训练稳定性。第二，动态场景已经覆盖低电量、高负载、紧急任务、高延迟和正常场景，但场景参数仍主要来自仿真配置，后续需要进一步贴近真实机器人任务流和通信链路。第三，消融实验仍需补充，包括去除GAT、去除节点打分、去除场景识别、去除动态奖励、去除充电机制、调整启发式门控等。'
Set-ParagraphText 71 '第四，当前结果显示PPO-GAT-Scoring在主场景、四类专门场景和mixed-context场景中均优于PPO-Baseline和PPO-GAT-Naive，但仍需扩大随机种子数量、增加置信区间或统计显著性检验，使论文结论更严谨。第五，轻量系统验证已经跑通，但尚未接入Gazebo/ROS2或更真实的多机器人仿真平台，系统级说服力还需要增强。第六，论文写作素材已有雏形，但方法公式化、实验图表解释、相关工作归纳和创新点凝练仍需继续完善。'
Set-ParagraphText 73 '尚未完成的工作主要包括：一是进一步扩大环境规模，将机器人数量、计算节点数量和任务类型数量逐步提升，验证算法在规模变化下的稳定性；二是补充传统算法和启发式算法对比，包括先来先服务、最短作业优先、MinMin、MaxMin或负载均衡类策略；三是补充消融实验，包括去除GAT、去除节点打分、去除启发式门控、去除场景识别、去除动态奖励、去除充电机制和去除拥塞传播等；四是完善动态权重奖励函数的数学表达和公式化描述；五是接入或模拟更接近实际机器人系统的通信过程与任务执行过程；六是整理论文正文和实验图表。'
Set-ParagraphText 75 '针对规模不足问题，下一阶段将在不改变核心环境接口的情况下继续扩展配置，形成20机器人10节点、24机器人10节点、20机器人12节点以及更大规模场景，并保持相同评估脚本输出可比指标。针对动态场景问题，将继续完善mixed-context配置，使正常、低电量、高负载、紧急任务、高延迟/低带宽和充电恢复过程在一次训练中同时出现，并通过inferred_scenario和场景得分字段检验自动识别效果。'
Set-ParagraphText 76 '针对消融不足问题，将基于现有训练脚本增加ablation配置，保持训练步数、随机种子和环境参数一致，仅改变一个模块，以避免对比混杂。针对动态权重问题，将把当前实现中的场景识别、权重调节和奖励组成进一步公式化，明确不同上下文因子对能耗、deadline、过载、队列、延迟和负载均衡权重的影响。针对系统验证不足问题，短期内继续增强轻量系统验证的事件日志和可视化，长期将与ROS2/Gazebo流程对接，使任务产生、机器人状态和调度决策之间形成更真实的数据交互。'
Set-ParagraphText 78 '2026年6月：完成现有20机器人/10节点主实验、四类专门场景和mixed-context实验的结果整理；补充至少一组更大规模场景；完善评估脚本，使其自动输出均值、标准差、置信区间、场景得分和节点选择分布；修正已有代码中部分注释乱码和结果字段命名不统一问题。'
Set-ParagraphText 79 '2026年7月：完成传统调度算法补充和消融实验第一轮，重点比较PPO-Baseline、PPO-GAT-Scoring、无节点打分GAT、无启发式门控GAT、无动态奖励和传统启发式策略；输出实验表格和柱状图，形成论文实验章节初稿。'
Set-ParagraphText 80 '2026年8月：完善动态权重奖励函数和多场景实验，重点分析正常、紧急、低电量、高负载、高延迟和充电恢复场景下策略的权衡能力；补充失败案例分析，说明算法边界和改进方向。'
Set-ParagraphText 81 '2026年9月：开展更大规模重复实验和结果整理，固定最终模型参数，完成关键图表、统计分析和实验结论。根据实验结果确定论文主方法和对比方法的最终表述。'

# Update date
Set-ParagraphText 9 '填报日期：2026年5月26日'

$settings = New-Object System.Xml.XmlWriterSettings
$settings.Encoding = New-Object System.Text.UTF8Encoding($false)
$settings.Indent = $false
$writer = [System.Xml.XmlWriter]::Create($xmlPath, $settings)
$doc.Save($writer)
$writer.Close()
