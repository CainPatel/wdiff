"""wdiff command line: `wdiff structural BASE EDITED --out DIR`, then `wdiff directional --out DIR`."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from wdiff import __version__
from wdiff.load import CheckpointMismatch, iter_pairs, resolve
from wdiff.structural import diff, format_report, rollup


def run_structural(args: argparse.Namespace) -> int:
    started = datetime.now(UTC).isoformat()
    base, edited = resolve(args.base), resolve(args.edited)
    try:
        records = diff(iter_pairs(base.path, edited.path))
    except CheckpointMismatch as e:
        print(e, file=sys.stderr)
        return 2
    summary = rollup(records)
    report = {
        "meta": {
            "tool": "wdiff",
            "version": __version__,
            "stage": "structural",
            "started": started,
            "finished": datetime.now(UTC).isoformat(),
            "base": {"source": base.source, "revision": base.revision},
            "edited": {"source": edited.source, "revision": edited.revision},
            "args": {k: v for k, v in vars(args).items() if k != "func"},
        },
        "summary": summary,
        "tensors": records,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "structural.json").write_text(json.dumps(report, indent=2))
    print(format_report(records, summary, args.top))
    print(f"\nwrote {out / 'structural.json'}")
    return 0


def run_directional(args: argparse.Namespace) -> int:
    from wdiff import directional as d

    started = datetime.now(UTC).isoformat()
    out = Path(args.out)
    structural = json.loads((out / "structural.json").read_text())
    meta = structural["meta"]
    base, edited = resolve(meta["base"]["source"]), resolve(meta["edited"]["source"])
    names = [t["name"] for t in structural["tensors"] if t["changed"] > 0]
    records = [
        d.tensor_direction(n, d.read_tensor(base.path, n), d.read_tensor(edited.path, n), args.rank)
        for n in names
    ]
    cos = d.cosine_matrix(records)
    overlap = d.subspace_overlap(records)
    named = []
    if args.directions:
        from wdiff.directions import load_named, named_cosines

        named = named_cosines(records, load_named(Path(args.directions)))
    for r in records:
        del r["_basis"]
    report = {
        "meta": {
            "tool": "wdiff",
            "version": __version__,
            "stage": "directional",
            "started": started,
            "finished": datetime.now(UTC).isoformat(),
            "base": meta["base"],
            "edited": meta["edited"],
            "energy_threshold": d.ENERGY,
            "args": {k: v for k, v in vars(args).items() if k != "func"},
        },
        "tensors": records,
        "direction_cosines": cos,
        "subspace_overlap": overlap,
        "named_direction_cosines": named,
    }
    (out / "directional.json").write_text(json.dumps(report, indent=2))
    print(d.format_report(records, cos) if records else "no changed tensors")
    if named:
        print("\ncosine with named directions (sign is arbitrary); energy inside edit subspace:")
        for row in named:
            print(
                f"  {row['tensor']}  vs  {row['direction']}[{row['layer']}]: {row['cosine']:+.4f}"
                f"   in_subspace {row['in_subspace']:.4f} (random {row['random_baseline']:.4f})"
            )
    print(f"\nwrote {out / 'directional.json'}")
    return 0


def run_direction_compute(args: argparse.Namespace) -> int:
    from wdiff.directions import compute

    base = resolve(args.base)
    layers = [int(x) for x in args.layers.split(",")]
    r = compute(base.source, base.path, base.revision, layers, Path(args.out))
    print(f"refusal direction {list(r.shape)} for layers {layers} -> {args.out}")
    return 0


def run_behavioral(args: argparse.Namespace) -> int:
    from wdiff import behavioral as bh

    started = datetime.now(UTC).isoformat()
    out = Path(args.out)
    base, edited = resolve(args.base), resolve(args.edited)
    probes = bh.load_probes(Path(args.probes), args.max_prompts)
    device = bh.pick_device(args.device)
    total = 2 * sum(len(rows) for rows in probes.values())
    budget = None if args.force else 90 * 60
    print(f"{total} generations on {device}, max_new_tokens={args.max_new_tokens}")

    gens, scores, tok_len, thinking = {}, {}, None, {}
    for label, ckpt in (("base", base), ("edited", edited)):
        print(f"{label}: {ckpt.source}")
        runner = bh.Runner(ckpt.path, device)
        tok_len, thinking[label] = runner.tok_len, runner.thinking
        try:
            gens[label] = bh.run_model(
                runner, probes, args.max_new_tokens, args.batch_size, args.seed,
                budget if label == "base" else None, total,
            )  # fmt: skip
        except bh.BudgetExceeded as e:
            print(e, file=sys.stderr)
            return 3
        finally:
            runner.close()
        scores[label] = {
            name: [bh.score(g, r.get("answer"), tok_len) for g, r in zip(gs, probes[name], strict=True)]
            for name, gs in gens[label].items()
        }  # fmt: skip

    metrics = {
        name: {label: bh.aggregate(scores[label][name]) for label in ("base", "edited")}
        for name in probes
    }
    for m in metrics.values():
        m["delta"] = {k: m["edited"][k] - m["base"][k] for k in m["base"] if k != "n"}

    samples = out / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for name, rows in probes.items():
        with (samples / f"{name}.jsonl").open("w") as f:
            for i, r in enumerate(rows):
                rec = {**r}
                for label in ("base", "edited"):
                    g = gens[label][name][i]
                    rec[label] = {"text": g["text"], **scores[label][name][i]}
                f.write(json.dumps(rec) + "\n")

    report = {
        "meta": {
            "tool": "wdiff",
            "version": __version__,
            "stage": "behavioral",
            "started": started,
            "finished": datetime.now(UTC).isoformat(),
            "base": {"source": base.source, "revision": base.revision},
            "edited": {"source": edited.source, "revision": edited.revision},
            "device": device,
            "generation": {**bh.SETTINGS, "thinking": thinking, "samples_per_prompt": 1,
                           "max_new_tokens": args.max_new_tokens, "seed": args.seed},
            "loop": {"window": bh.WINDOW, "repeats": bh.REPEATS},
            "refusal_phrases": bh.REFUSAL_PHRASES,
            "args": {k: v for k, v in vars(args).items() if k != "func"},
        },
        "metrics": metrics,
    }  # fmt: skip
    (out / "behavioral.json").write_text(json.dumps(report, indent=2))
    print(format_behavioral(metrics))
    print(f"\nwrote {out / 'behavioral.json'} and {samples}/")
    return 0


def run_summarize(args: argparse.Namespace) -> int:
    from wdiff import summarize as sm

    started = datetime.now(UTC).isoformat()
    out = Path(args.out)
    ctx = sm.build_context(out)
    sm.check_no_weights(ctx)
    message, tokens = sm.summarize(ctx, args.model)
    invented, shape = sm.check_numbers(message, ctx), sm.check_shape(message)
    record = {
        "meta": {"tool": "wdiff", "version": __version__, "stage": "summarize",
                 "started": started, "finished": datetime.now(UTC).isoformat(),
                 "model": args.model, "input_tokens": tokens,
                 "args": {k: v for k, v in vars(args).items() if k != "func"}},
        "context": ctx,
        "message": message,
        "invented_numbers": invented,
        "shape_problems": shape,
    }  # fmt: skip
    (out / "summarize.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(message)
    print(f"\ninput {tokens} tokens")
    if invented or shape:
        (out / "COMMIT.rejected.md").write_text(message + "\n")
        print(f"REJECTED: invented numbers {invented}; shape {shape}", file=sys.stderr)
        print(f"wrote {out / 'COMMIT.rejected.md'}; COMMIT.md not written", file=sys.stderr)
        return 4
    (out / "COMMIT.md").write_text(message + "\n")
    print(f"every number found in the reports; wrote {out / 'COMMIT.md'}")
    return 0


def run_all(args: argparse.Namespace) -> int:
    """structural -> refusal direction (if missing) -> directional -> behavioral -> summarize."""
    ns = argparse.Namespace
    out, dirs = Path(args.out), Path(args.directions)
    if rc := run_structural(ns(base=args.base, edited=args.edited, out=args.out, top=20)):
        return rc
    if not (dirs / "refusal.npy").exists():
        per_layer = json.loads((out / "structural.json").read_text())["summary"]["per_layer"]
        layers = ",".join(k for k, v in per_layer.items() if k.isdigit() and v["changed"] > 0)
        rc = run_direction_compute(ns(base=args.base, layers=layers, out=str(dirs)))
        if rc:
            return rc
    steps = [
        (run_directional, ns(out=args.out, rank=8, directions=str(dirs))),
        (run_behavioral, ns(base=args.base, edited=args.edited, out=args.out, probes=args.probes,
                            max_new_tokens=args.max_new_tokens, max_prompts=args.max_prompts,
                            device=args.device, batch_size=args.batch_size, seed=args.seed,
                            force=args.force)),
        (run_summarize, ns(out=args.out, model=args.model)),
    ]  # fmt: skip
    for fn, a in steps:
        if rc := fn(a):
            return rc
    return 0


def format_behavioral(metrics: dict) -> str:
    keys = ["loop_rate", "hit_cap_rate", "repeat_rate", "answer_rate", "accuracy",
            "refusal_rate", "short_rate", "no_response_rate", "mean_tokens"]  # fmt: skip
    lines = []
    for name, m in metrics.items():
        lines += [
            "",
            f"{name} (n={m['base']['n']})",
            f"  {'metric':<17}{'base':>9}{'edited':>9}{'delta':>9}",
        ]
        for k in keys:
            if k in m["base"]:
                fmt = ".0f" if k == "mean_tokens" else ".2f"
                lines.append(
                    f"  {k:<17}{m['base'][k]:>9{fmt}}{m['edited'][k]:>9{fmt}}{m['delta'][k]:>+9{fmt}}"
                )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="wdiff", description="Semantic diff of two checkpoints")
    sub = p.add_subparsers(required=True)
    s = sub.add_parser("structural", help="which tensors changed and by how much")
    s.add_argument("base", help="local directory or Hugging Face repo id")
    s.add_argument("edited", help="local directory or Hugging Face repo id")
    s.add_argument("--out", required=True)
    s.add_argument("--top", type=int, default=20)
    s.set_defaults(func=run_structural)
    s = sub.add_parser("directional", help="rank and direction of each changed tensor")
    s.add_argument("--out", required=True, help="report dir holding structural.json")
    s.add_argument("--rank", type=int, default=8, help="singular values to keep")
    s.add_argument("--directions", help="dir with <name>.npy + provenance.json to compare against")
    s.set_defaults(func=run_directional)
    s = sub.add_parser("direction", help="named directions").add_subparsers(required=True)
    s = s.add_parser("compute", help="compute a named direction for a base model")
    s.add_argument("base", help="local directory or Hugging Face repo id")
    s.add_argument("--method", choices=["refusal"], default="refusal")
    s.add_argument("--layers", required=True, help="comma-separated, e.g. 15,19")
    s.add_argument("--out", required=True)
    s.set_defaults(func=run_direction_compute)
    s = sub.add_parser("behavioral", help="what the two models do differently on probe prompts")
    s.add_argument("base", help="local directory or Hugging Face repo id")
    s.add_argument("edited", help="local directory or Hugging Face repo id")
    s.add_argument("--out", required=True)
    s.add_argument("--probes", default="probes")
    s.add_argument("--max-new-tokens", type=int, default=1024)
    s.add_argument("--max-prompts", type=int, default=50, help="per probe set")
    s.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true", help="skip the 90-minute budget check")
    s.set_defaults(func=run_behavioral)
    s = sub.add_parser("summarize", help="commit message from the three reports")
    s.add_argument("--out", required=True, help="report dir with the three JSON files")
    s.add_argument("--model", default="claude-opus-5-5")
    s.set_defaults(func=run_summarize)
    s = sub.add_parser("all", help="every stage, then the commit message")
    s.add_argument("base", help="local directory or Hugging Face repo id")
    s.add_argument("edited", help="local directory or Hugging Face repo id")
    s.add_argument("--out", required=True)
    s.add_argument("--directions", required=True, help="dir for (or holding) refusal.npy")
    s.add_argument("--probes", default="probes")
    s.add_argument("--max-new-tokens", type=int, default=1024)
    s.add_argument("--max-prompts", type=int, default=50)
    s.add_argument("--device", default="auto")
    s.add_argument("--batch-size", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.add_argument("--model", default="claude-opus-5-5")
    s.set_defaults(func=run_all)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
