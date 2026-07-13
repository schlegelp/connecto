# Installation

connecto needs **Python 3.11 or newer**.

=== "pip"

    ```bash
    pip install connecto
    ```

=== "uv"

    ```bash
    uv add connecto
    ```

=== "From source"

    ```bash
    git clone https://github.com/schlegelp/connecto
    cd connecto
    pip install -e ".[dev]"
    ```

## What comes with it

These are hard dependencies, not extras:

| package | why |
|---|---|
| [`caveclient`](https://github.com/CAVEconnectome/CAVEclient) | the CAVE backend — FlyWire, BANC, MICrONS |
| [`neuprint-python`](https://connectome-neuprint.github.io/neuprint-python/) | the neuPrint backend — hemibrain, maleCNS, MANC, fish2 |
| [`navis`](https://navis-org.github.io/navis/) | skeletons and meshes come back as navis neurons |
| [`cloud-volume`](https://github.com/seung-lab/cloud-volume) | meshes, and CAVE's skeleton service reaches for it internally |
| `pandas`, `numpy`, `pyarrow`, `networkx`, `trimesh`, `tqdm` | the usual |

!!! warning "caveclient must be ≥ 8.0"

    This is pinned for a reason. Before 8.0, caveclient streamed query results as
    CSV and re-inferred the types on the way back in — which silently truncated
    int64 root IDs. 8.0 switched to a pandas-native path. If you have an older
    caveclient pinned by something else in your environment, connecto's root IDs
    can come back subtly wrong rather than obviously broken.

## Optional extras

Some annotation sources are lab-internal or need an extra package:

```bash
pip install "connecto[clio]"      # Clio annotations (maleCNS)
pip install "connecto[flytable]"  # SeaTable / "flytable" annotations
pip install "connecto[points]"    # parallel point -> segment lookups
```

Nothing in the core depends on these. If you ask for a source you have not
installed, you get a `MissingDependencyError` telling you the exact `pip install`
to run — not an `ImportError` from three frames down.

## Check it worked

```python
import connecto as cn

cn.list_datasets()
```

```
                 name                          label                  species        backends                     annotations
0                banc    BANC (brain and nerve cord)  Drosophila melanogaster  cave, neuprint                  cave, flytable
1               fish2       fish2 (larval zebrafish)              Danio rerio        neuprint                        neuprint
2             flywire  FlyWire (FAFB) public release  Drosophila melanogaster  cave, neuprint                public, flytable
3  flywire-production      FlyWire (FAFB) production  Drosophila melanogaster            cave                public, flytable
4           hemibrain                      hemibrain  Drosophila melanogaster        neuprint                        neuprint
5             malecns                       male CNS  Drosophila melanogaster        neuprint                  neuprint, clio
6                manc                MANC (male VNC)  Drosophila melanogaster        neuprint                        neuprint
7             microns      MICrONS (minnie65) public             Mus musculus            cave  celltypes, mtypes, nucleus_svm
8          optic-lobe                     optic lobe  Drosophila melanogaster        neuprint                        neuprint
```

That call is offline — it only reads the registry. To actually *query* anything you
need tokens, which is the [next page](credentials.md).
