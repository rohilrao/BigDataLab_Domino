from domino import explore, DominoSlicer
import meerkat as mk
import torch

DEVICE = 0

print(torch.cuda.is_available())

import os

dp = mk.datasets.get("imagenette", dataset_dir="C:/Users/rohil/Downloads/Uni Bonn/WiSe 2022-23/BigDataLab_Domino/BigDataLab_Domino/src/meerkat-main/meerkat/datasets/imagenet/")

# we'll only be using the validation data
dp = dp.lz[dp["split"] == "valid"]

print(dp.shape)

import torch
from torchvision.models import resnet18
import torchvision.transforms as transforms
model = resnet18(pretrained=True)

# 1. Define transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

# 2. Create new column with transform
dp["input"] = dp["img"].to_lambda(transform)

# 1. Move the model to device
model.to(DEVICE).eval()

# 2. Define a function that runs a forward pass over a batch
@torch.no_grad()
def predict(batch: mk.DataPanel):
    input_col: mk.TensorColumn = batch["input"]
    x: torch.Tensor = input_col.data.to(DEVICE)  # We get the underlying torch tensor with `data` and move to GPU
    out: torch.Tensor = model(x)  # Run forward pass

    # Return a dictionary with one key for each of the new columns. Each value in the
    # dictionary should have the same length as the batch.
    return {
        "pred": out.cpu().numpy().argmax(axis=-1),
        "probs": torch.softmax(out, axis=-1).cpu().numpy(),
    }

# 3. Apply the update. Note that the `predict` function operates on batches, so we set
# `is_batched_fn=True`. Also, the `predict` function only accesses the "input" column, by
# specifying that here we instruct update to only load that one column and skip others
dp = dp.update(
    function=predict,
    is_batched_fn=True,
    batch_size=32,
    input_columns=["input"],
    pbar=True
)

dp["correct"] = dp["pred"] == mk.NumpyArrayColumn(dp["label_idx"])
accuracy = dp["correct"].mean()
print(f"Micro accuracy across the ten Imagenette classes: {accuracy:0.3}")


LABEL_IDX = 571

# convert to a binary task
dp["prob"] = dp["probs"][:, LABEL_IDX]
dp["target"] = (dp["label_idx"] == LABEL_IDX)

from domino import embed
dp = embed(
    dp,
    input_col="img",
    encoder="clip",
    device=DEVICE,
    num_workers=0
)

domino = DominoSlicer(
    y_log_likelihood_weight=40,
    y_hat_log_likelihood_weight=40,
    n_mixture_components=25,
    n_slices=5
)

domino.fit(data=dp, embeddings="clip(img)", targets="target", pred_probs="prob")

dp["domino_slices"] = domino.predict_proba(
    data=dp, embeddings="clip(img)", targets="target", pred_probs="prob"
)


from domino import generate_candidate_descriptions
phrase_templates = [
    "a photo of [MASK].",
    "a photo of {} [MASK].",
    "a photo of [MASK] {}.",
    "a photo of [MASK] {} [MASK].",
]

text_dp = generate_candidate_descriptions(
    templates=phrase_templates,
    num_candidates=10_000
)

from domino import describe

dp["target"] = dp["target"].astype(int)

descriptions = describe(
    data=dp,
    embeddings="clip(img)",
    targets="target",
    slices="domino_slices",
    text=text_dp,
    text_embeddings="clip(output_phrase)",
    slice_idx=0
)
descriptions.lz[(-descriptions["score"]).argsort()][:10]