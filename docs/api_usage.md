# API Usage

Main entry points:

- `operadic_consistency.run_consistency_check`
- `operadic_consistency.run_consistency_check_from_question`

Minimal pattern:

```python
from operadic_consistency import ToQ, ToQNode, run_consistency_check

# define ToQ
# implement Answerer and Collapser callables
# call run_consistency_check(...)
```

For a complete runnable example, see `examples/minimal_consistency.py`.
