# Selecting neurons

```python
ds.ids(720575940604407468)          # an ID
ds.ids("DA1_lPN")                   # a type
ds.ids("/^AOTU00.*")                # a regex
ds.ids("super_class:visual_projection")   # any annotation column
ds.ids("DA1_lPN", side="left")
ds.ids(NeuronCriteria(type="DA1_lPN", side="left"))
```

The first five desugar into the sixth. See the
[tutorial](../tutorials/selecting-neurons.md) for the guided version.

::: connecto.core.criteria
    options:
      members:
        - NeuronCriteria
        - parse_ids
        - is_id
