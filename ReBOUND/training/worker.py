import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["periodicity", "grouped", "residual", "foundation"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="target", choices=["target", "source_a", "source_b", "source_c"])
    parser.add_argument("--component", choices=["diag", "off"])
    parser.add_argument("--recipe", choices=["regularized", "wide", "conservative"])
    parser.add_argument("--group", choices=["top", "middle", "rare"])
    parser.add_argument("--loss", choices=["mse", "tweedie"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--scope", type=int)
    parser.add_argument("--foundation-group", type=int, choices=range(4))
    parser.add_argument("--model-path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-memory-gib", type=float, default=2.)
    args = parser.parse_args()
    if args.kind == "foundation":
        from .foundation import train_foundation_group
        train_foundation_group(args.root, args.output, args.foundation_group, args.model_path,
                                args.device, args.gpu_memory_gib)
    elif args.kind == "residual":
        from .residual import fit_scope
        fit_scope(args.root, args.output, args.scope)
    else:
        from .experts import train_grouped, train_periodicity
        if args.kind == "periodicity":
            train_periodicity(args.root, args.output, args.task, args.component, args.recipe,
                              args.seed, args.device, args.gpu_memory_gib)
        else:
            train_grouped(args.root, args.output, args.task, args.group, args.loss,
                           args.seed, args.device, args.gpu_memory_gib)


if __name__ == "__main__":
    main()
