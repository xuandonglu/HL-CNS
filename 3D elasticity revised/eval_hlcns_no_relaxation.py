from model_hlcns import HLCNSNoRelaxation3D
from eval_hlcns_no_correction import eval_hlcns_no_correction


def eval_hlcns_no_relaxation(
    data_dir: str,
    model_path: str,
    best_model_path: str | None = None,
    problem_type: str = "elasticity",
    batch_size: int = 10,
    n_cycles: int = 1,
    max_levels: int = 3,
    E: float = 1.0,
    nu: float = 0.3,
    kappa: float = 1.0,
):
    return eval_hlcns_no_correction(
        data_dir=data_dir,
        model_path=model_path,
        best_model_path=best_model_path,
        model_cls=HLCNSNoRelaxation3D,
        problem_type=problem_type,
        batch_size=batch_size,
        n_cycles=n_cycles,
        max_levels=max_levels,
        E=E,
        nu=nu,
        kappa=kappa,
    )


if __name__ == "__main__":
    mode = "nonlinear_twopeak"
    n = 1
    l = 3
    for i in range(1, 6):
        data_dir = f"./data_{mode}"
        model_dir = f"model_{mode}/hlcns_no_relaxation"
        eval_hlcns_no_relaxation(
            data_dir=data_dir,
            model_path=f"{model_dir}/checkpoint_hlcns_no_relaxation_n{n}_l{l}_{i}.pt",
            best_model_path=f"{model_dir}/best_hlcns_no_relaxation_n{n}_l{l}_{i}.pt",
            problem_type="elasticity",
            batch_size=10,
            E=1.0,
            nu=0.3,
            n_cycles=n,
            max_levels=l,
        )
