import numpy as np

def old_eta2(Ea_Pa, Ta_K):
    eta1 = 0.465 * Ea_Pa / Ta_K
    eta2_arg = 1.2 + 3 * eta1
    # Return nan for negative argument to mimic new implementation (avoiding complex numbers)
    if eta2_arg >= 0:
        return -np.sqrt(eta2_arg)
    else:
        return np.nan

def new_eta2(Ea_Pa, Ta_K):
    eta1 = 0.465 * Ea_Pa / Ta_K
    eta2_arg = 1.2 + 3 * eta1
    return np.where(eta2_arg >= 0, -np.sqrt(eta2_arg), np.nan)

Ea_Pa_values = np.array([0, 1, 5, 10, 20])
Ta_K_values = np.array([270, 280, 290, 300, 310])

print(f"{'Ea_Pa':>6} {'Ta_K':>6} {'old_eta2':>15} {'new_eta2':>15} {'match':>8}")
for Ea_Pa in Ea_Pa_values:
    for Ta_K in Ta_K_values:
        old_val = old_eta2(Ea_Pa, Ta_K)
        new_val = new_eta2(Ea_Pa, Ta_K)
        # new_val is an array even for single values, so take the item
        new_val_item = new_val.item() if hasattr(new_val, 'item') else new_val
        if np.isnan(old_val) and np.isnan(new_val_item):
            match = True
        else:
            match = np.allclose(old_val, new_val_item, equal_nan=True)
        print(f"{Ea_Pa:6} {Ta_K:6} {old_val:15} {new_val_item:15} {str(match):>8}")