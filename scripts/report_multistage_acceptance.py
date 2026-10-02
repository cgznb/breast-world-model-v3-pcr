"""Map every supplied T01-T60 requirement to executed evidence or a stated gap."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from responsewm.io import read_json,write_json


MAPPING={
1:["test_patient_split_disjoint"],2:["test_stats_train_only"],3:["test_known_at_filter_and_labels_absent"],
4:["test_four_masks_not_conflated"],5:["test_nonadjacent_pair_tagging"],6:["test_stage_ids_not_observation_count"],
7:["test_four_stage_curriculum_finite_and_checkpoint_schema"],8:["test_request_rejects_codec_identity_and_actual_future_dates"],
9:["test_canonical_grid_query_subset_and_readonly"],10:["test_canonical_grid_query_subset_and_readonly"],
11:["test_current_query_is_repeatable_and_no_future_signature","test_five_state_apis_and_cli_without_future_assets"],
12:["test_noise_batch_order_and_sample_identity","test_patient_singleton_noise_agrees_with_batch"],
13:["test_noise_batch_order_and_sample_identity"],14:["test_independent_request_has_no_future_file_access"],
15:["test_supervision_poison_does_not_change_v2_forecast"],16:["test_free_rollout_three_hops_and_missing_truth"],
17:["test_observation_seed_has_image_gradient"],18:["test_observation_anchor_idempotence_conflict_and_invalidation"],
19:["test_observe_preserves_history_and_teacher_coordinates"],20:["test_clinical_payload_change_is_not_idempotent","test_observation_anchor_idempotence_conflict_and_invalidation"],
21:["test_observation_anchor_idempotence_conflict_and_invalidation"],22:["test_online_observation_only_replay_parity"],
23:["test_invalid_observation_leaves_whole_state_unchanged"],24:["test_shared_head_and_observation_can_change_risk"],
25:["test_history_block_causal_outputs_are_stable"],26:["test_future_interval_plan_does_not_change_earlier_physiology"],
27:["test_history_block_causal_outputs_are_stable"],28:["test_symmflow_endpoints_and_directions","test_training_horizon_is_explicit_and_not_marginal"],
29:["test_request_rejects_codec_identity_and_actual_future_dates"],31:["test_encoder_pyramid_preserves_default_graph_and_keys"],
32:["test_zero_bridge_preserves_native_image_and_parameter_layout"],33:["test_early_bridge_changes_deeper_grid_and_decoder"],
34:["test_marginal_bce_reaches_both_stream_parameters"],35:["test_multihop_gradient_reaches_first_state"],
36:["test_deep_jepa_stopgrad_targets_and_fixed_teacher_input_gradients"],37:["test_checkpointed_causal_rollout_matches_gradients"],
38:["test_checkpoint_forward_backward_and_training_determinism"],39:["test_monai_v2_zero_bridge_split_matches_actual_backend"],
40:["test_shared_head_and_observation_can_change_risk"],41:["test_marginal_is_probability_mean_not_logit_mean","test_extreme_logits_are_stable"],
42:["test_zero_label_support_has_zero_pcr_gradient"],43:["test_energy_never_mixes_patients","test_independent_time_sample_permutation_changes_joint_energy"],
44:["test_stage_and_joint_distribution_have_distinct_terms"],45:["test_sampler_expectation_coverage_and_exact_resume"],
46:["test_sampler_expectation_coverage_and_exact_resume"],47:["test_supervision_poison_does_not_change_v2_forecast","test_free_rollout_three_hops_and_missing_truth"],
48:["test_observed_update_trains_continuation_from_real_anchor"],50:["test_deep_jepa_stopgrad_targets_and_fixed_teacher_input_gradients"],
51:["test_four_stage_training_and_input_only_deployment","test_checkpoint_resume_exact_and_mutation_guard"],
52:["test_migration_maps_semantic_roles_and_preserves_new_state_modules","test_required_shape_mismatch_rejects_without_partial_mutation"],
53:["test_step_boundary_resume_exact","test_joint_step_boundary_resume_exact"],54:["test_independent_request_rejects_target_and_identity"],
55:["test_probability_metrics_and_cluster_ci"],56:["test_five_state_apis_and_cli_without_future_assets"],
57:["test_v2_configs_register_and_validate"],59:["test_four_stage_curriculum_finite_and_checkpoint_schema"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--junit",nargs="+",required=True);p.add_argument("--preflight",required=True)
    p.add_argument("--output",required=True);args=p.parse_args()
    tests={};suites=[]
    for filename in args.junit:
        tree=ET.parse(filename)
        counts={"passed":0,"failed":0,"skipped":0}
        for case in tree.findall(".//testcase"):
            status="failed" if case.find("failure") is not None or case.find("error") is not None else "skipped" if case.find("skipped") is not None else "tested"
            counts[{"tested":"passed","failed":"failed","skipped":"skipped"}[status]]+=1
            name=case.attrib["name"].split("[")[0]
            tests.setdefault(name,[]).append({"status":status,"junit":filename,"test":case.attrib["name"]})
        suites.append({"junit":filename,**counts})
    rows=[]
    for number in range(1,61):
        names=MAPPING.get(number,[])
        evidence=[entry for name in names for entry in tests.get(name,[])]
        all_pass=bool(names) and all(any(v["status"]=="tested" for v in tests.get(name,[])) for name in names)
        rows.append({"id":f"T{number:02}","status":"tested" if all_pass else "not-run","evidence":evidence})
    rows[29].update(status="skipped",reason="KV cache deliberately disabled; full raw-prefix recomputation is the implementation.")
    rows[48].update(status="not-run",reason="Loss implementation reviewed: no risk/volume monotonicity or prior=posterior penalty. No dedicated automated monotonicity regression.")
    preflight=read_json(args.preflight)
    verified=set(preflight.get("results",{}))=={"representation","flow","readout","joint"}
    rows[57].update(status="tested" if verified else "not-run",evidence=[{"preflight":args.preflight,"stages":list(preflight.get("results",{}))}])
    rows[21]["scope"]="Exact online/replay parity for observations_only protocol; predictive-prior protocol separately tested by prequential data-read ordering."
    rows[52]["scope"]="Exact CPU deterministic A and D resume; CUDA exact equality is not claimed."
    rows[54]["scope"]="Patient bootstrap arithmetic and distinct observed-stage reporting, not external-cohort clinical calibration."
    assert len(rows)==60 and not any(s["failed"] for s in suites)
    rows[59].update(status="tested",evidence=[{"generator":str(Path(__file__).name),"checked":"60 rows plus executed JUnit counts; unsupported items explicitly recorded"}])
    write_json(args.output,{"schema":"multistage_acceptance_v2","suites":suites,"acceptance":rows,
                           "clinical_performance_validated":False,"all_spec_items_automatically_passed":False})


if __name__=="__main__":
    main()
