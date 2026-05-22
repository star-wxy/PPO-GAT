$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

& (Join-Path $scriptRoot "run_ppo_model_scenario.ps1") `
    -EnvConfig "configs/scenarios/mixed_context_20r_10n.yaml" `
    -BaselineTrainConfig "configs/scenarios/train_plain_ppo_mixed_context_20r_10n.yaml" `
    -NaiveTrainConfig "configs/scenarios/train_naive_gat_mixed_context_20r_10n.yaml" `
    -ScoringTrainConfig "configs/scenarios/train_scoring_gat_mixed_context_20r_10n.yaml" `
    -OutputPrefix "scenario_mixed_context_ppo_models_dynamic"
