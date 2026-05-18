import wandb

api = wandb.Api()
run = api.run("/anandhuprakash070-bits-pilani/AquaCLR-LEGION-M1/runs/zt7p87w5")

print(run.history())
