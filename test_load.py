
import megatron_het
print("Vendor info:", megatron_het.get_vendor_info())
print("Module location:", megatron_het.__file__)

print(megatron_het.core)
print(megatron_het.training)

from megatron_het.training import get_args
