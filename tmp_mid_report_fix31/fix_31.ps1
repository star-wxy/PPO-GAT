$base = 'D:\Mytest\tmp_mid_report_fix31'
$xmlPath = Join-Path $base 'unzipped\word\document.xml'
[xml]$doc = Get-Content -LiteralPath $xmlPath -Encoding UTF8
$ns = New-Object System.Xml.XmlNamespaceManager($doc.NameTable)
$ns.AddNamespace('w','http://schemas.openxmlformats.org/wordprocessingml/2006/main')
$body = $doc.SelectSingleNode('//w:body', $ns)
$paras = $doc.SelectNodes('//w:body/w:p', $ns)
$wNs = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

function Set-ParagraphText {
    param([System.Xml.XmlNode]$P, [string]$Text)
    $runs = @($P.SelectNodes('./w:r', $ns))
    foreach ($r in $runs) { [void]$P.RemoveChild($r) }
    $rNew = $doc.CreateElement('w','r',$wNs)
    $tNew = $doc.CreateElement('w','t',$wNs)
    $xmlSpace = $doc.CreateAttribute('xml','space','http://www.w3.org/XML/1998/namespace')
    $xmlSpace.Value = 'preserve'
    [void]$tNew.Attributes.Append($xmlSpace)
    $tNew.InnerText = $Text
    [void]$rNew.AppendChild($tNew)
    [void]$P.AppendChild($rNew)
}

function Copy-ParagraphProperties {
    param([System.Xml.XmlNode]$Target, [System.Xml.XmlNode]$Like)
    $targetPr = $Target.SelectSingleNode('./w:pPr', $ns)
    if ($targetPr) { [void]$Target.RemoveChild($targetPr) }
    $likePr = $Like.SelectSingleNode('./w:pPr', $ns)
    if ($likePr) {
        $imported = $doc.ImportNode($likePr, $true)
        [void]$Target.PrependChild($imported)
    }
}

function New-ParagraphLike {
    param([System.Xml.XmlNode]$Like, [string]$Text)
    $newP = $Like.CloneNode($true)
    Set-ParagraphText -P $newP -Text $Text
    return $newP
}

# Current paragraph layout in the updated report:
# 54 = 2.9 title, 55/56 were mistakenly occupied by 3.1 body text, 57 = 3.2 title, 68 = level-1 title style.
$p29Body = $paras[55]
$pChapterStyle = $paras[68]
$pSubheadingStyle = $paras[57]
$pBodyStyle = $paras[55]

Set-ParagraphText -P $p29Body -Text '截至本次中期检查，暂未完成论文发表或专利申请。当前主要工作集中在算法实现、仿真环境搭建、动态奖励机制设计和实验数据积累。下一阶段计划在补充消融实验、统计显著性分析和更大规模场景验证后，整理形成论文投稿材料或技术报告。'

$pChapter = $paras[56]
Copy-ParagraphProperties -Target $pChapter -Like $pChapterStyle
Set-ParagraphText -P $pChapter -Text '三、关键技术或难点（创新点）'

$ref = $pChapter
$newTexts = @(
    @{ Like = $pSubheadingStyle; Text = '3.1 《算力-任务-机器人》协同建模' },
    @{ Like = $pBodyStyle; Text = '已经完成的第一项关键技术是面向算力互联网场景的《算力-任务-机器人》协同建模。与只考虑计算节点负载或单一任务卸载的调度模型不同，本研究将机器人画像、任务类型、计算节点资源、网络拓扑关系和系统运行状态放在同一仿真环境中统一描述。机器人不再只是任务集合，而是具有本地CPU、剩余电量、home节点、任务到达率、任务规模偏好、deadline偏好和队列状态的动态实体；任务不再是无差别作业，而是具有来源机器人、任务类型、任务规模、优先级、deadline、本地计算需求和传输需求的结构化对象；算力节点也不是静态服务器，而是具有CPU容量、基础时延、能耗因子、实时负载和邻接拥塞传播影响的异构资源。' },
    @{ Like = $pBodyStyle; Text = '这一建模方式使智能体能够在每一步同时感知机器人侧和算力侧的状态变化，并学习调度动作对多目标指标的综合影响。例如，选择云端节点可能获得更高CPU容量，但会带来更大的网络时延、拓扑传输代价和能耗；选择本地或近邻边缘节点可以降低通信代价，但在节点负载较高时可能造成排队和过载；当机器人电量下降并进入充电状态时，系统还需要在任务继续产生、低电量保护和节点选择之间重新权衡。通过这种统一建模，论文方法能够支撑PPO-GAT-Scoring、上下文感知动态奖励和多场景自动识别机制，使调度策略不只优化单个任务的即时完成时间，而是面向多机器人持续任务流学习更稳定的全局折中策略。' }
)

foreach ($item in $newTexts) {
    $newP = New-ParagraphLike -Like $item.Like -Text $item.Text
    [void]$body.InsertAfter($newP, $ref)
    $ref = $newP
}

$settings = New-Object System.Xml.XmlWriterSettings
$settings.Encoding = New-Object System.Text.UTF8Encoding($false)
$settings.Indent = $false
$writer = [System.Xml.XmlWriter]::Create($xmlPath, $settings)
$doc.Save($writer)
$writer.Close()
