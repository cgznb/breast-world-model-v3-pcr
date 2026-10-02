"""Input-only deployment and codec operations with a recorded numeric contract."""
from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from .data import ManifestStore,exact_keys,validate_input
from .training import load_trained
from .legacy.codec import load_codec
from .io import read_json,write_json,digest,autocast,seed_all


def read_request(path,payload):
    request = read_json(path)
    keys = {"schema","clinical_features","action_features","phase_order","latent_shape","vq_identity","time_basis","input"}
    exact_keys(request,keys,keys)
    if request["schema"] != "responsewm_request_v1":
        raise ValueError("Prediction accepts an input-only request, never a training manifest")
    contract = payload["metadata"]["data_contract"]
    for key,value in contract.items():
        if request[key] != value:
            raise ValueError(f"Deployment {key} differs from the checkpoint")
    # A source-only reader: no cases, no target arrays, no fitting methods invoked.
    reader = ManifestStore.__new__(ManifestStore)
    reader.path = Path(path).resolve()
    reader.manifest = request
    reader.c,reader.a = len(request["clinical_features"]),len(request["action_features"])
    reader.statistics = payload["metadata"]["statistics"]
    reader.cache = OrderedDict(); reader.cache_size = 0
    validate_input(request["input"],reader.c,reader.a)
    if request["time_basis"] == "stage_index":
        coordinates = [v["day"] for v in request["input"]["observed"]]+[q["day"] for q in request["input"]["queries"]]
        if any(v != int(v) for v in coordinates):
            raise ValueError("Stage-index query coordinates must be integers")
    return reader.normalized_input([request["input"]]),request


@torch.no_grad()
def predict(checkpoint,request_path,output,*,device="cpu",samples=None,steps=None,seed=0,codec_path=None):
    model,payload = load_trained(checkpoint,device)
    if not bool(model.readout_ready or model.joint_ready):
        raise ValueError("This checkpoint has no trained generated-trajectory pCR readout")
    seed_all(seed,model.cfg.training.threads)
    inp,request = read_request(request_path,payload)
    inp = inp.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    with autocast(device,model.cfg.training.precision):
        result = model.forecast(inp,samples=samples,steps=steps,generator=generator)
    # Convert continuous VQ coordinates back before ANY decoding.
    mean = model.encoder.latent_mean[None,None]
    std = model.encoder.latent_std[None,None]
    raw = result.latent*std+mean
    arrays = {"latent":raw.float().cpu().numpy(),"state":result.state.float().cpu().numpy(),
              "image_state":result.image_state.float().cpu().numpy(),
              "trajectory_probabilities":result.logits.float().sigmoid().cpu().numpy(),
              "pcr_probability":result.probability.cpu().numpy(),"future_days":inp.future_days.cpu().numpy()}
    if codec_path:
        identity = "sha256:"+digest(codec_path)
        if request["vq_identity"] != identity:
            raise ValueError("Codec digest differs from VQ identity; refusing mismatched decoder")
        codec = load_codec(codec_path,device)
        shape = raw.shape
        flat = raw.reshape(-1,*shape[-4:])
        if len(flat):
            decoded = torch.cat([codec.decode(z[None]).cpu() for z in flat],0)
            arrays["images"] = decoded.reshape(*shape[:3],*decoded.shape[1:]).numpy()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)
    info = {"checkpoint_sha256":digest(checkpoint),"request_sha256":digest(request_path),"seed":seed,
            "samples":result.logits.shape[1],"steps":steps or model.cfg.sampling.inference_steps,
            "latent_coordinates":"raw continuous VQ coordinates","probability_aggregation":"mean of per-trajectory probabilities",
            "clinical_validation":False,"synthetic_checkpoint":payload["metadata"]["synthetic"],
            "images_decoded":codec_path is not None,"shape":{k:list(v.shape) for k,v in arrays.items()}}
    write_json(Path(output).with_suffix(".json"),info)
    return info


@torch.no_grad()
def encode_images(images_path,codec_path,output,*,device="cpu",normalization_record):
    """Already registered, cropped and intensity-normalized three-phase MRI only.

    The normalization JSON is retained as provenance. This function does NOT
    guess a normalization transform or claim to preprocess raw scanner DICOM.
    """
    record = read_json(normalization_record)
    if record.get("already_normalized") is not True or record.get("source_only_geometry") is not True:
        raise ValueError("Declare prior VQ-matched intensity normalization and source-only geometry")
    if record.get("vq_sha256") != digest(codec_path):
        raise ValueError("Normalization record names a different codec")
    images = np.load(images_path,allow_pickle=False)
    if isinstance(images,np.lib.npyio.NpzFile):
        with images as values:
            images = np.asarray(values["images"],np.float32)
    if images.ndim == 4:
        images = images[None]
    if images.ndim != 5 or images.shape[0] != 1 or images.shape[1] != 3 or not np.isfinite(images).all():
        raise ValueError("Encode one finite three-phase visit per file: [3,D,H,W] or [1,3,D,H,W]")
    codec = load_codec(codec_path,device)
    z = codec.encode(torch.as_tensor(images,dtype=torch.float32,device=device)).cpu().numpy()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,latent=z[0] if len(z) == 1 else z)
    write_json(Path(output).with_suffix(".json"),{"vq_identity":"sha256:"+digest(codec_path),
               "normalization_record":record,"latent_representation":"continuous_pre_quantization",
               "shape":list(z.shape),"phase_order":["pre_aqc0","first_post_aqc1","metadata_late"]})
