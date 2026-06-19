from commodity_strategy import (
    build_production_strategy_weights,
    print_latest_allocation,
)


def main():
    weights = build_production_strategy_weights(
        save_final_scores=True,
        save_target_weights=True,
    )

    print_latest_allocation(weights)

    return weights


if __name__ == "__main__":
    main()