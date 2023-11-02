from tqdm import tqdm
import time , motley

# Specify a custom bar string formatting. May impact performance.
# '{l_bar}{bar}{r_bar}'
# where
# l_bar='{desc}: {percentage:3.0f}%|'
# r_bar='|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'

# Possible vars: 
# l_bar, bar, r_bar, n, n_fmt, total, total_fmt, percentage, elapsed,
# elapsed_s, ncols, nrows, desc, unit, rate, rate_fmt, rate_noinv,
# rate_noinv_fmt, rate_inv, rate_inv_fmt, postfix, unit_divisor, remaining,
# remaining_s, eta.

# Note that a trailing ": " is automatically removed after {desc} if the latter
# is empty.

fmt = motley.stylize(
    '{{desc}: {percentage:3.0f}%{bar}'
    '{n_fmt}/{total_fmt}:|lime} '
    '{rate_fmt:|gold} '
    '{elapsed:|cyan} eta {remaining:|cyan}'
)
for _ in tqdm(range(200), unit=' frames', #colour='#006400',
              ascii=' ╸━', bar_format=fmt):
    time.sleep(0.05)
