"""Real author-model GPU/CPU forward/backward check; no agricultural scores."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from cropnet_leo.remote import write_json
from .mmst_bridge import ROOT, make_model
from .mmst_training import seed_all, vector


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((ROOT/'research/joint_fusion/configs/mmst_compare_balanced_v1.json').read_text())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_all(10)
    start = time.perf_counter()
    model = make_model(config).to(device)
    initial = vector(model)
    inputs = [torch.randn(1,6,1,3,224,224,device=device), torch.randn(1,6,1,28,9,device=device), torch.randn(1,5,12,9,device=device)]
    model.eval()
    with torch.no_grad():
        model.batch_size = 6
        reference = model(inputs[0], ys=inputs[1], yl=inputs[2])
        model.batch_size = 5  # Forces a final singleton batch in the author PVT.
        chunked = model(inputs[0], ys=inputs[1], yl=inputs[2])
        torch.testing.assert_close(reference, chunked, atol=2e-5, rtol=2e-5)
    model.batch_size = config['image_chunk_size']
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    output = model(inputs[0], ys=inputs[1], yl=inputs[2])
    loss = (output - output.new_tensor([[10., 4.]])).square().mean()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
    optimizer.step()
    final = vector(model)
    assert not torch.equal(initial, final)
    report = {'status': 'passed', 'scope': 'synthetic interface check only; no crop-yield performance result',
              'parameters': initial.numel(), 'device': device,
              'device_name': torch.cuda.get_device_name() if device == 'cuda' else 'CPU',
              'output_shape': list(output.shape), 'all_used_gradients_finite': True,
              'unused_parameter_names': [name for name,p in model.named_parameters() if p.grad is None],
              'singleton_chunk_max_output_difference': float((reference-chunked).abs().max()),
              'max_parameter_update': float((final-initial).abs().max()),
              'initial_sha256': hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
              'final_sha256': hashlib.sha256(final.numpy().tobytes()).hexdigest(),
              'elapsed_seconds': time.perf_counter()-start}
    if args.out.exists():
        raise FileExistsError(args.out)
    write_json(args.out, report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
