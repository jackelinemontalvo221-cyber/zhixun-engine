import random

import swanlab


random.seed(42)

swanlab.init(
    project="swanlab-smoke-test",
    name="connectivity-check",
    description="Check that SwanLab can record experiment metrics.",
    config={
        "purpose": "connectivity-test",
        "python_version": "3.15.2",
    },
    tags=["smoke-test"],
)

for step in range(1, 6):
    accuracy = 0.5 + step * 0.05 + random.random() * 0.01
    loss = 1.0 / (step + 1) + random.random() * 0.01

    swanlab.log(
        {
            "simulated/accuracy": accuracy,
            "simulated/loss": loss,
        },
        step=step,
    )

swanlab.finish()
print("SwanLab smoke test uploaded.")
