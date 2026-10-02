#!/usr/bin/env python
"""Report optional backend availability without silently installing anything."""
import importlib.util,json,platform,torch
result={'python':platform.python_version(),'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
        'monai_installed':importlib.util.find_spec('monai') is not None}
if result['monai_installed']:
    import monai
    result['monai_version']=monai.__version__
    result['monai_pinned_match']=monai.__version__=='1.5.1'
if torch.cuda.is_available():
    result.update(device=torch.cuda.get_device_name(),bf16_supported=torch.cuda.is_bf16_supported())
print(json.dumps(result,indent=2))
