"""
Generate validation analysis assets for PromptIR.

Outputs:
- analysis/metrics.csv
- analysis/prompt_weights.csv
- analysis/degradation_wise_metrics.csv
- analysis/summary.json
- analysis/visuals/*.png
- analysis/prompt_plots/*.png
"""
import argparse
import csv
import json
import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import random_split

from dataset import ImageRestorationDataset
from losses import SSIMLoss
from metrics import calculate_psnr
from promptir_model import PromptIR


IMAGE_SIZE = (256, 256)
MODEL_BASE_DIM = 48
MODEL_NUM_BLOCKS_PER_LEVEL = [3, 4, 4, 6]
MODEL_NUM_REFINEMENT_BLOCKS = 4
MODEL_NUM_PROMPT_COMPONENTS = 5
MODEL_PG_PROMPT_DIM_MAP = {0: 256, 1: 128, 2: 64}
MODEL_PG_BASE_HW_MAP = {
    0: IMAGE_SIZE[0] // 16,
    1: IMAGE_SIZE[0] // 8,
    2: IMAGE_SIZE[0] // 4,
}
MODEL_BACKBONE_ATTN_HEADS = 8
MODEL_PROMPT_INTERACTION_ATTN_HEADS = 8


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_state_dict(path, device):
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if all(key.startswith("module.") for key in state_dict.keys()):
        stripped = OrderedDict()
        for key, value in state_dict.items():
            stripped[key[7:]] = value
        state_dict = stripped
    return state_dict


def infer_checkpoint_features(state_dict):
    keys = tuple(state_dict.keys())
    return {
        "use_degradation_classifier": any(
            key.startswith("degradation_classifier.") for key in keys
        ),
        "use_task_prompt_bank": any(
            ".shared_prompt_components" in key
            or ".rain_prompt_components" in key
            or ".snow_prompt_components" in key
            for key in keys
        ),
        "use_frequency_branch": any(
            key.startswith("frequency_branch.") or key.startswith("frequency_fusion.")
            for key in keys
        ),
    }


def resolve_model_features(
    detected_features,
    use_degradation_classifier=False,
    use_task_prompt_bank=False,
    use_frequency_branch=False,
):
    requested_features = {
        "use_degradation_classifier": use_degradation_classifier,
        "use_task_prompt_bank": use_task_prompt_bank,
        "use_frequency_branch": use_frequency_branch,
    }
    resolved_features = {
        name: requested_features[name] or detected_features[name]
        for name in requested_features
    }

    auto_enabled = [
        name.replace("use_", "").replace("_", "-")
        for name, enabled in resolved_features.items()
        if enabled and detected_features[name] and not requested_features[name]
    ]
    if auto_enabled:
        print("Auto-enabled checkpoint features: " + ", ".join(auto_enabled))

    strict_load = not any(
        requested_features[name] and not detected_features[name]
        for name in requested_features
    )
    return resolved_features, strict_load


def build_model(
    checkpoint_path,
    device,
    use_degradation_classifier=False,
    use_task_prompt_bank=False,
    use_frequency_branch=False,
):
    state_dict = load_state_dict(checkpoint_path, device)
    detected_features = infer_checkpoint_features(state_dict)
    model_features, strict_load = resolve_model_features(
        detected_features,
        use_degradation_classifier=use_degradation_classifier,
        use_task_prompt_bank=use_task_prompt_bank,
        use_frequency_branch=use_frequency_branch,
    )
    model = PromptIR(
        in_channels=3,
        out_channels=3,
        base_dim=MODEL_BASE_DIM,
        num_blocks_per_level=MODEL_NUM_BLOCKS_PER_LEVEL,
        num_refinement_blocks=MODEL_NUM_REFINEMENT_BLOCKS,
        num_prompt_components=MODEL_NUM_PROMPT_COMPONENTS,
        pg_prompt_dim_map=MODEL_PG_PROMPT_DIM_MAP,
        pg_base_hw_map=MODEL_PG_BASE_HW_MAP,
        backbone_num_attn_heads=MODEL_BACKBONE_ATTN_HEADS,
        prompt_interaction_num_attn_heads=MODEL_PROMPT_INTERACTION_ATTN_HEADS,
        use_degradation_classifier=model_features["use_degradation_classifier"],
        use_task_prompt_bank=model_features["use_task_prompt_bank"],
        use_frequency_branch=model_features["use_frequency_branch"],
        bias=False,
    )
    load_result = model.load_state_dict(state_dict, strict=strict_load)
    if load_result.missing_keys:
        print("Missing checkpoint keys:")
        for key in load_result.missing_keys:
            print(f"  {key}")
    if load_result.unexpected_keys:
        print("Unexpected checkpoint keys:")
        for key in load_result.unexpected_keys:
            print(f"  {key}")
    model.to(device)
    model.eval()
    return model


def extract_restored(model_output):
    if isinstance(model_output, dict):
        return model_output["restored"]
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    return model_output


def build_validation_subset(degraded_dir, clean_dir, val_ratio, seed):
    dataset = ImageRestorationDataset(
        degraded_base_dir=degraded_dir,
        clean_base_dir=clean_dir,
        patch_size=IMAGE_SIZE[0],
        is_train=False,
    )
    total = len(dataset)
    if total == 0:
        raise RuntimeError("No validation candidates found. Check Data/train/degraded and Data/train/clean.")

    val_size = max(1, int(val_ratio * total))
    train_size = total - val_size
    _, val_subset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    return val_subset


def tensor_to_uint8_image(tensor):
    array = torch.clamp(tensor.detach().cpu(), 0.0, 1.0).numpy()
    array = np.transpose(array, (1, 2, 0))
    return (array * 255.0 + 0.5).astype(np.uint8)


def error_to_uint8_image(a, b):
    diff = torch.mean(torch.abs(a.detach().cpu() - b.detach().cpu()), dim=0).numpy()
    diff = np.clip(diff * 4.0, 0.0, 1.0)
    heat = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = (diff * 255).astype(np.uint8)
    heat[..., 1] = (np.sqrt(diff) * 180).astype(np.uint8)
    heat[..., 2] = ((1.0 - diff) * 40).astype(np.uint8)
    return heat


def add_label(image, label):
    font = ImageFont.load_default()
    pad = 6
    label_h = 20
    canvas = Image.new("RGB", (image.width, image.height + label_h), "white")
    canvas.paste(image, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 5), label, fill=(20, 20, 20), font=font)
    return canvas


def make_comparison(degraded, restored, clean, output_path, title):
    panels = [
        add_label(Image.fromarray(tensor_to_uint8_image(degraded)), "degraded"),
        add_label(Image.fromarray(tensor_to_uint8_image(restored)), "restored"),
        add_label(Image.fromarray(tensor_to_uint8_image(clean)), "clean"),
        add_label(Image.fromarray(error_to_uint8_image(restored, clean)), "error x4"),
    ]
    margin = 8
    title_h = 24
    width = sum(panel.width for panel in panels) + margin * (len(panels) - 1)
    height = max(panel.height for panel in panels) + title_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill=(20, 20, 20), font=ImageFont.load_default())
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, title_h))
        x += panel.width + margin
    canvas.save(output_path)


def make_prompt_plot(prompt_rows, output_path, title):
    if not prompt_rows:
        return
    bar_w = 34
    group_gap = 28
    top = 42
    bottom = 34
    max_h = 140
    width = len(prompt_rows) * (MODEL_NUM_PROMPT_COMPONENTS * bar_w + group_gap) + group_gap
    height = top + max_h + bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = [(49, 130, 189), (230, 85, 13), (49, 163, 84), (117, 107, 177), (99, 99, 99)]
    draw.text((8, 8), title, fill=(20, 20, 20), font=font)

    x = group_gap
    baseline = top + max_h
    for row in prompt_rows:
        for i, weight in enumerate(row["weights"]):
            bar_h = int(float(weight) * max_h)
            x0 = x + i * bar_w
            draw.rectangle((x0, baseline - bar_h, x0 + bar_w - 6, baseline), fill=colors[i])
            draw.text((x0, baseline + 4), f"p{i}", fill=(40, 40, 40), font=font)
        draw.text((x, baseline + 18), row["stage"], fill=(40, 40, 40), font=font)
        x += MODEL_NUM_PROMPT_COMPONENTS * bar_w + group_gap
    canvas.save(output_path)


def get_prompt_rows(model):
    stage_names = ["deep", "mid", "shallow"]
    rows = []
    for idx, prompt_gen in enumerate(model.prompt_gens):
        weights = prompt_gen.last_prompt_weights
        if weights is None:
            continue
        rows.append({
            "stage": stage_names[idx] if idx < len(stage_names) else f"stage_{idx}",
            "weights": weights[0].detach().cpu().tolist(),
        })
    return rows


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average(values):
    return float(sum(values) / len(values)) if values else 0.0


def infer_degradation_type(filename):
    name = filename.lower()
    if "rain" in name:
        return "rain"
    if "snow" in name:
        return "snow"
    return "unknown"


def aggregate_group_metrics(records, group_name=None):
    if group_name is None:
        group_records = records
    else:
        group_records = [
            record for record in records
            if record["degradation"] == group_name
        ]

    return {
        "num_images": len(group_records),
        "avg_input_psnr": average([record["input_psnr"] for record in group_records]),
        "avg_restored_psnr": average([record["restored_psnr"] for record in group_records]),
        "avg_psnr_gain": average([record["psnr_gain"] for record in group_records]),
        "avg_ssim": average([record["ssim"] for record in group_records]),
    }


def build_degradation_wise_metrics(records):
    return {
        "rain": aggregate_group_metrics(records, "rain"),
        "snow": aggregate_group_metrics(records, "snow"),
        "all": aggregate_group_metrics(records, None),
    }


def write_degradation_wise_csv(path, degradation_wise_metrics):
    rows = []
    for degradation in ["rain", "snow", "all"]:
        metrics = degradation_wise_metrics[degradation]
        rows.append({
            "degradation": degradation,
            "num_images": metrics["num_images"],
            "avg_input_psnr": f"{metrics['avg_input_psnr']:.4f}",
            "avg_restored_psnr": f"{metrics['avg_restored_psnr']:.4f}",
            "avg_psnr_gain": f"{metrics['avg_psnr_gain']:.4f}",
            "avg_ssim": f"{metrics['avg_ssim']:.6f}",
        })
    write_csv(
        path,
        rows,
        [
            "degradation", "num_images", "avg_input_psnr",
            "avg_restored_psnr", "avg_psnr_gain", "avg_ssim",
        ],
    )


def print_degradation_wise_table(degradation_wise_metrics):
    print("\n=== Degradation-wise Performance ===")
    print(f"{'Type':<6} {'Num':>5} {'Input PSNR':>12} {'Restored PSNR':>15} {'Gain':>9} {'SSIM':>9}")
    for degradation in ["rain", "snow", "all"]:
        metrics = degradation_wise_metrics[degradation]
        print(
            f"{degradation.capitalize():<6} "
            f"{metrics['num_images']:>5} "
            f"{metrics['avg_input_psnr']:>12.4f} "
            f"{metrics['avg_restored_psnr']:>15.4f} "
            f"{metrics['avg_psnr_gain']:>9.4f} "
            f"{metrics['avg_ssim']:>9.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze PromptIR validation behavior.")
    parser.add_argument("--checkpoint", default="trained_models/promptir_best.pth")
    parser.add_argument("--degraded-dir", default="Data/train/degraded")
    parser.add_argument("--clean-dir", default="Data/train/clean")
    parser.add_argument("--output-dir", default="analysis")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-visuals", type=int, default=5)
    parser.add_argument("--use-degradation-classifier", action="store_true")
    parser.add_argument("--use-task-prompt-bank", action="store_true")
    parser.add_argument("--use-frequency-branch", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)
    visuals_dir = os.path.join(args.output_dir, "visuals")
    prompt_dir = os.path.join(args.output_dir, "prompt_plots")
    ensure_dir(visuals_dir)
    ensure_dir(prompt_dir)

    model = build_model(
        args.checkpoint,
        device,
        use_degradation_classifier=args.use_degradation_classifier,
        use_task_prompt_bank=args.use_task_prompt_bank,
        use_frequency_branch=args.use_frequency_branch,
    )
    val_subset = build_validation_subset(args.degraded_dir, args.clean_dir, args.val_ratio, args.seed)
    ssim_loss = SSIMLoss(data_range=1.0, channel=3).to(device)

    metrics_rows = []
    metric_records = []
    prompt_rows_all = []
    cached_outputs = []

    with torch.no_grad():
        for local_idx, dataset_idx in enumerate(val_subset.indices):
            degraded, clean, label = val_subset.dataset[dataset_idx]
            item = val_subset.dataset.file_list[dataset_idx]
            filename = os.path.basename(item["path"])

            degraded_b = degraded.unsqueeze(0).to(device)
            clean_b = clean.unsqueeze(0).to(device)
            restored_b = torch.clamp(extract_restored(model(degraded_b)), 0.0, 1.0)

            restored = restored_b[0].detach().cpu()
            clean_cpu = clean_b[0].detach().cpu()
            degraded_cpu = degraded_b[0].detach().cpu()

            psnr = calculate_psnr(restored, clean_cpu)
            input_psnr = calculate_psnr(degraded_cpu, clean_cpu)
            psnr_gain = psnr - input_psnr
            ssim = 1.0 - float(ssim_loss(restored_b, clean_b).item())
            input_ssim = 1.0 - float(ssim_loss(degraded_b, clean_b).item())
            l1 = float(torch.mean(torch.abs(restored - clean_cpu)).item())
            degradation = infer_degradation_type(filename)

            row = {
                "filename": filename,
                "label": int(label.item()) if hasattr(label, "item") else int(label),
                "psnr": f"{psnr:.4f}",
                "ssim": f"{ssim:.6f}",
                "l1": f"{l1:.6f}",
                "input_psnr": f"{input_psnr:.4f}",
                "input_ssim": f"{input_ssim:.6f}",
                "improvement_psnr": f"{psnr_gain:.4f}",
            }
            metrics_rows.append(row)
            metric_records.append({
                "filename": filename,
                "degradation": degradation,
                "input_psnr": float(row["input_psnr"]),
                "restored_psnr": float(row["psnr"]),
                "psnr_gain": float(row["improvement_psnr"]),
                "ssim": float(row["ssim"]),
            })

            prompt_rows = get_prompt_rows(model)
            for prompt_row in prompt_rows:
                prompt_csv_row = {
                    "filename": filename,
                    "label": row["label"],
                    "stage": prompt_row["stage"],
                }
                for i, weight in enumerate(prompt_row["weights"]):
                    prompt_csv_row[f"prompt_{i}"] = f"{float(weight):.6f}"
                prompt_rows_all.append(prompt_csv_row)

            cached_outputs.append({
                "filename": filename,
                "psnr": psnr,
                "degraded": degraded_cpu,
                "restored": restored,
                "clean": clean_cpu,
                "prompt_rows": prompt_rows,
            })

            if (local_idx + 1) % 10 == 0:
                print(f"Analyzed {local_idx + 1}/{len(val_subset)} images")

    metric_fields = [
        "filename", "label", "psnr", "ssim", "l1",
        "input_psnr", "input_ssim", "improvement_psnr",
    ]
    prompt_fields = ["filename", "label", "stage"] + [f"prompt_{i}" for i in range(MODEL_NUM_PROMPT_COMPONENTS)]
    write_csv(os.path.join(args.output_dir, "metrics.csv"), metrics_rows, metric_fields)
    write_csv(os.path.join(args.output_dir, "prompt_weights.csv"), prompt_rows_all, prompt_fields)

    sorted_outputs = sorted(cached_outputs, key=lambda x: x["psnr"])
    worst = sorted_outputs[:args.num_visuals]
    best = sorted_outputs[-args.num_visuals:][::-1]
    step = max(1, len(sorted_outputs) // max(1, args.num_visuals))
    sampled = sorted_outputs[::step][:args.num_visuals]

    for group_name, group in [("best", best), ("worst", worst), ("sample", sampled)]:
        for rank, item in enumerate(group, start=1):
            stem = os.path.splitext(item["filename"])[0]
            comparison_path = os.path.join(visuals_dir, f"{group_name}_{rank:02d}_{stem}.png")
            make_comparison(
                item["degraded"],
                item["restored"],
                item["clean"],
                comparison_path,
                f"{group_name} #{rank}: {item['filename']} PSNR={item['psnr']:.2f}",
            )
            prompt_path = os.path.join(prompt_dir, f"{group_name}_{rank:02d}_{stem}_prompts.png")
            make_prompt_plot(item["prompt_rows"], prompt_path, f"{item['filename']} prompt weights")

    psnrs = [float(row["psnr"]) for row in metrics_rows]
    ssims = [float(row["ssim"]) for row in metrics_rows]
    input_psnrs = [float(row["input_psnr"]) for row in metrics_rows]
    improvements = [float(row["improvement_psnr"]) for row in metrics_rows]
    degradation_wise_metrics = build_degradation_wise_metrics(metric_records)
    summary = {
        "checkpoint": args.checkpoint,
        "num_images": len(metrics_rows),
        "avg_psnr": average(psnrs),
        "avg_ssim": average(ssims),
        "avg_input_psnr": average(input_psnrs),
        "avg_improvement_psnr": average(improvements),
        "degradation_wise_metrics": degradation_wise_metrics,
        "best_cases": [item["filename"] for item in best],
        "worst_cases": [item["filename"] for item in worst],
    }
    write_degradation_wise_csv(
        os.path.join(args.output_dir, "degradation_wise_metrics.csv"),
        degradation_wise_metrics,
    )
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_degradation_wise_table(degradation_wise_metrics)
    print(f"Saved analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
