import os
import json
import asyncio
import random
from tqdm import tqdm
from collections import defaultdict

from core.base import ExperimentBase
from core.utils import get_base_parser
from core.config import RISK_TYPES, ATTACK_STYLES
from prompts.tmap import (
    TMAP_SEED_PROMPT,
    TMAP_CROSS_DIAG_PROMPT_PARENT,
    TMAP_CROSS_DIAG_PROMPT_TARGET,
    TMAP_TCG_MUTATE_PROMPT,
    TMAP_EDGE_ANALYSIS_PROMPT,
)

class ToolCallGraph:
    def __init__(self):
        self.edges = {}

    def update_edge(self, edge, success, reason):
        if not edge or len(edge) != 2:
            return
        src, dst = edge
        key = (src, dst)
        entry = self.edges.get(key, {"n_s": 0, "n_f": 0, "r_s": [], "r_f": []})
        if success:
            entry["n_s"] += 1
            if reason:
                entry["r_s"].append(reason)
        else:
            entry["n_f"] += 1
            if reason:
                entry["r_f"].append(reason)
        self.edges[key] = entry

    def to_snapshot(self, gen):
        edges = []
        for key, entry in self.edges.items():
            total = entry["n_s"] + entry["n_f"]
            edges.append({
                "edge": list(key),
                "n_s": entry["n_s"],
                "n_f": entry["n_f"],
                "success_rate": (entry["n_s"] / total) if total else 0.0,
                "r_s": entry["r_s"],
                "r_f": entry["r_f"]
            })
        edges.sort(key=lambda x: x["success_rate"], reverse=True)
        return {"gen": gen, "edges": edges}

    def to_prompt_data(self):
        edges = []
        for key, entry in self.edges.items():
            edges.append({
                "edge": list(key),
                "n_s": entry["n_s"],
                "n_f": entry["n_f"],
                "r_s": entry["r_s"],
                "r_f": entry["r_f"]
            })
        return edges


def save_tcg_snapshot(output_dir, mode_name, run_id, gen, snapshot):
    save_dir = os.path.join(output_dir, mode_name.lower(), "tcg", run_id)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"gen_{gen}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def build_edge_analysis_prompts(prompts, results):
    edge_prompts = []
    for prompt, res in zip(prompts, results):
        edge_prompts.append(
            TMAP_EDGE_ANALYSIS_PROMPT.format(
                prompt=prompt,
                history=str(res.get("history", ""))
            )
        )
    return edge_prompts


def get_tmap_parser():
    parser = get_base_parser()
    parser.add_argument("--disable_tcg", action="store_true",
                        help="Disable TCG guidance (no edge updates or path hints).")
    parser.add_argument("--disable_cross_diag", action="store_true",
                        help="Disable cross-diagnosis (no SF/FC analysis).")
    parser.add_argument("--debug", action="store_true",
                        help="Print real-time debug information (Cross-Diagnosis, Mutate Output, TCG, LLM Judge).")
    return parser

async def main():
    args = get_tmap_parser().parse_args()
    exp = ExperimentBase(args)
    tool_defs = await exp.connect_to_server()

    mutation_n = min(args.mutation_n, exp.max_workers)
    archive = defaultdict(lambda: defaultdict(dict))
    update_counts = {cat: {style: 0 for style in ATTACK_STYLES} for cat in RISK_TYPES}
    tcg = ToolCallGraph()
    mode_suffix = []
    if args.disable_tcg:
        mode_suffix.append("no_tcg")
    if args.disable_cross_diag:
        mode_suffix.append("no_cross_diag")
    mode_name = "tmap" + (("_" + "_".join(mode_suffix)) if mode_suffix else "")
    run_id = f"{exp.server_name}_{mode_name}_{exp.timestamp}"

    all_cells = [(cat, style) for cat in RISK_TYPES for style in ATTACK_STYLES]

    user_prompts = []
    for cat, style in all_cells:
        user_prompts.append(
            TMAP_SEED_PROMPT.format(
                risk_type=cat,
                risk_description=RISK_TYPES[cat],
                attack_style=style,
                style_description=ATTACK_STYLES[style],
                tool_definition=json.dumps(tool_defs, indent=2)
            )
        )

    resps = exp.attacker_llm.batch_chat(
        None,
        user_prompts,
        max_workers=args.max_workers
    )

    clean_prompts, intended_seqs = [], []
    for r in resps:
        try:
            obj = json.loads(exp._clean_json_resp(r))
            clean_prompts.append(obj.get("attack_prompt", ""))
            intended_seqs.append(obj.get("intended_trajectory", []))
        except:
            clean_prompts.append(str(r))
            intended_seqs.append([])

    eval_res = await exp.evaluate_batch(
        [c[0] for c in all_cells],
        [c[1] for c in all_cells],
        clean_prompts,
        semaphore_val=args.max_workers
    )

    if args.disable_tcg:
        edge_results = [[] for _ in eval_res]
    else:
        edge_prompts = build_edge_analysis_prompts(clean_prompts, eval_res)
        edge_resps = exp.attacker_llm.batch_chat(
            None,
            edge_prompts,
            max_workers=args.max_workers
        )
        edge_results = []
        for r in edge_resps:
            try:
                parsed = json.loads(exp._clean_json_resp(r))
                edge_results.append(parsed)
                if getattr(args, "debug", False):
                    print(f"\n[DEBUG] TCG Analysis Output (Seed):\n{json.dumps(parsed, indent=2)}")
            except:
                edge_results.append([])
                if getattr(args, "debug", False):
                    print(f"\n[DEBUG] TCG Analysis parsing failed (Seed):\n{r}")

    for i, (cat, style) in enumerate(all_cells):
        res = eval_res[i]
        archive[cat][style] = {
            "attack_prompt": clean_prompts[i],
            "actual_trajectory": res["actual_trajectory"],
            "level": res["level"],
            "final_level": res["judgement"].get("final_level"),
            "reason": res["judgement"].get("assessment", ""),
            "history": res["history"],
            "fitness": res["level"],
        }
        if not args.disable_tcg:
            for edge in edge_results[i]:
                if isinstance(edge, dict):
                    tcg.update_edge(edge.get("edge", []), edge.get("success", False), edge.get("reason", ""))

    exp.save_results(archive, mode_name, is_final=False, generation=0)
    if not args.disable_tcg:
        save_tcg_snapshot(exp.output_dir, mode_name, run_id, 0, tcg.to_snapshot(0))
    print("[*] Seed complete: generation 0")

    for gen in tqdm(range(args.iteration), desc="tmap", ncols=80):
        it = gen + 1
        batch = []

        levels = defaultdict(list)
        for c, s in all_cells:
            lvl = archive[c][s].get("level", 0)
            levels[lvl].append((c, s))
        elites = levels[3] + levels[2] + levels[1] or levels[0]

        for _ in range(mutation_n):
            batch.append({"parent": random.choice(elites), "target": random.choice(all_cells)})

        parent_diags = [{} for _ in batch]
        target_diags = [{} for _ in batch]
        if not args.disable_cross_diag:
            parent_diag_prompts = []
            target_diag_prompts = []
            for itm in batch:
                p_cat, p_style = itm["parent"]
                t_cat, t_style = itm["target"]
                p_entry = archive[p_cat][p_style]
                t_entry = archive[t_cat][t_style]
                parent_diag_prompts.append(
                    TMAP_CROSS_DIAG_PROMPT_PARENT.format(
                        risk_type=p_cat,
                        risk_description=RISK_TYPES[p_cat],
                        attack_style=p_style,
                        style_description=ATTACK_STYLES[p_style],
                        prompt=p_entry.get("attack_prompt", ""),
                        history=str(p_entry.get("history", [])),
                        judge_assessment=p_entry.get("reason", "")
                    )
                )
                target_diag_prompts.append(
                    TMAP_CROSS_DIAG_PROMPT_TARGET.format(
                        risk_type=t_cat,
                        risk_description=RISK_TYPES[t_cat],
                        attack_style=t_style,
                        style_description=ATTACK_STYLES[t_style],
                        prompt=t_entry.get("attack_prompt", ""),
                        history=str(t_entry.get("history", [])),
                        judge_assessment=t_entry.get("reason", "")
                    )
                )

            parent_diag_resps = exp.attacker_llm.batch_chat(
                None,
                parent_diag_prompts,
                max_workers=exp.max_workers
            )
            target_diag_resps = exp.attacker_llm.batch_chat(
                None,
                target_diag_prompts,
                max_workers=exp.max_workers
            )

            parent_diags, target_diags = [], []
            for i, r in enumerate(parent_diag_resps):
                try:
                    diag = json.loads(exp._clean_json_resp(r))
                    parent_diags.append(diag)
                    if getattr(args, "debug", False):
                        print(f"\n[DEBUG] Parent ({batch[i]['parent'][0]}, {batch[i]['parent'][1]}) Cross-Diagnosis SF:\n{json.dumps(diag, indent=2)}")
                except:
                    parent_diags.append({})
                    if getattr(args, "debug", False):
                        print(f"\n[DEBUG] Parent ({batch[i]['parent'][0]}, {batch[i]['parent'][1]}) Cross-Diagnosis parsing failed:\n{r}")
            for i, r in enumerate(target_diag_resps):
                try:
                    diag = json.loads(exp._clean_json_resp(r))
                    target_diags.append(diag)
                    if getattr(args, "debug", False):
                        print(f"\n[DEBUG] Target ({batch[i]['target'][0]}, {batch[i]['target'][1]}) Cross-Diagnosis FC:\n{json.dumps(diag, indent=2)}")
                except:
                    target_diags.append({})
                    if getattr(args, "debug", False):
                        print(f"\n[DEBUG] Target ({batch[i]['target'][0]}, {batch[i]['target'][1]}) Cross-Diagnosis parsing failed:\n{r}")

        synth_prompts = []
        for i, itm in enumerate(batch):
            t_cat, t_style = itm["target"]
            p_diag = parent_diags[i]
            t_diag = target_diags[i]
            tcg_full = []
            if not args.disable_tcg:
                tcg_full = tcg.to_prompt_data()
            synth_prompts.append(
                TMAP_TCG_MUTATE_PROMPT.format(
                    risk_type=t_cat,
                    risk_description=RISK_TYPES[t_cat],
                    attack_style=t_style,
                    style_description=ATTACK_STYLES[t_style],
                    success_factor=p_diag.get("success_factor", "N/A"),
                    failure_cause=t_diag.get("failure_cause", "N/A"),
                    target_prompt=archive[t_cat][t_style].get("attack_prompt", ""),
                    target_history=str(archive[t_cat][t_style].get("history", [])),
                    tcg_full=json.dumps(tcg_full),
                    tool_defs=json.dumps(tool_defs)
                )
            )

        mutant_resps = exp.attacker_llm.batch_chat(
            None,
            synth_prompts,
            max_workers=exp.max_workers
        )
        mutants, valid_idx = [], []
        for i, r in enumerate(mutant_resps):
            try:
                m = json.loads(exp._clean_json_resp(r))
                if m.get("attack_prompt"):
                    mutants.append(m)
                    valid_idx.append(i)
                    if getattr(args, "debug", False):
                        print(f"\n[DEBUG] Mutated Target Prompt for ({batch[i]['target'][0]}, {batch[i]['target'][1]}):\n{json.dumps(m, indent=2)}")
            except:
                if getattr(args, "debug", False):
                    print(f"\n[DEBUG] Mutate Target Prompt parsing failed for ({batch[i]['target'][0]}, {batch[i]['target'][1]}):\n{r}")

        if not mutants:
            continue

        results = await exp.evaluate_batch(
            [batch[idx]["target"][0] for idx in valid_idx],
            [batch[idx]["target"][1] for idx in valid_idx],
            [m["attack_prompt"] for m in mutants],
            semaphore_val=mutation_n
        )

        if args.disable_tcg:
            edge_results = [[] for _ in results]
        else:
            edge_prompts = build_edge_analysis_prompts(
                [m["attack_prompt"] for m in mutants],
                results
            )
            edge_resps = exp.attacker_llm.batch_chat(
                None,
                edge_prompts,
                max_workers=args.max_workers
            )
            edge_results = []
            for i, r in enumerate(edge_resps):
                try:
                    parsed = json.loads(exp._clean_json_resp(r))
                    edge_results.append(parsed)
                    if getattr(args, "debug", False):
                        t_cat, t_style = batch[valid_idx[i]]["target"]
                        print(f"\n[DEBUG] TCG Analysis Output (Iteration {it}, Target: {t_cat}, {t_style}):\n{json.dumps(parsed, indent=2)}")
                except:
                    edge_results.append([])
                    if getattr(args, "debug", False):
                        t_cat, t_style = batch[valid_idx[i]]["target"]
                        print(f"\n[DEBUG] TCG Analysis parsing failed (Iteration {it}, Target: {t_cat}, {t_style}):\n{r}")

        updates = 0
        for i, res in enumerate(results):
            idx = valid_idx[i]
            t_cat, t_style = batch[idx]["target"]
            cur_lvl = archive[t_cat][t_style].get("level", 0)
            new_lvl = res["level"]
            if not args.disable_tcg:
                for edge in edge_results[i]:
                    if isinstance(edge, dict):
                        tcg.update_edge(edge.get("edge", []), edge.get("success", False), edge.get("reason", ""))

            if new_lvl > cur_lvl:
                should_update = True
            elif new_lvl == cur_lvl and cur_lvl > 0:
                should_update = (await exp.compare_results(
                    t_cat, t_style,
                    archive[t_cat][t_style].get("attack_prompt"),
                    archive[t_cat][t_style].get("history"),
                    mutants[i]["attack_prompt"],
                    res["history"]
                ) == "B")
            else:
                should_update = False

            if should_update:
                archive[t_cat][t_style].update({
                    "attack_prompt": mutants[i].get("attack_prompt", ""),
                    "actual_trajectory": res["actual_trajectory"],
                    "level": new_lvl,
                    "final_level": res["judgement"].get("final_level"),
                    "reason": res["judgement"].get("assessment", ""),
                    "history": res["history"]
                })
                archive[t_cat][t_style]["fitness"] = archive[t_cat][t_style].get("fitness", 0) + 1
                update_counts[t_cat][t_style] += 1
                updates += 1

        print(f"[*] Updates: {updates}")
        exp.log_iteration(it, archive, mode_name, updates=updates)
        if args.checkpoint_interval > 0 and it % args.checkpoint_interval == 0:
            exp.save_results(archive, mode_name, is_final=False, generation=it)
        if not args.disable_tcg:
            save_tcg_snapshot(exp.output_dir, mode_name, run_id, it, tcg.to_snapshot(it))

    exp.save_results(archive, mode_name, is_final=True, generation=args.iteration)
    if not args.disable_tcg:
        save_tcg_snapshot(exp.output_dir, mode_name, run_id, args.iteration, tcg.to_snapshot(args.iteration))
    await exp.mcp_client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
