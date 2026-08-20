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
| [`DracoPy`](https://github.com/seung-lab/DracoPy) | meshes are draco-encoded everywhere connecto reads them |
| `pandas`, `numpy`, `pyarrow`, `networkx`, `trimesh`, `tqdm` | the usual |

!!! note "connecto does not use cloud-volume"

    Segmentation volumes and meshes are read by `connecto.precomputed`, connecto's
    own reader for the neuroglancer precomputed and graphene formats. cloud-volume
    also *writes*, to Google Cloud, S3 and half a dozen other backends, and pays for
    that in dependencies — boto3, the Google Cloud SDK, gevent, protobuf and the
    rest come to roughly 78 MB across some 37 packages, none of which reading needs.

    connecto's reader is checked against cloud-volume rather than instead of it:
    `tests/test_precomputed.py` reads the same cutouts and the same meshes both ways
    and asserts they agree, so cloud-volume is a development dependency.

!!! warning "caveclient must be ≥ 8.0"

    This is pinned for a reason. Before 8.0, caveclient streamed query results as
    CSV and re-inferred the types on the way back in — which silently truncated
    int64 root IDs. 8.0 switched to a pandas-native path. If you have an older
    caveclient pinned by something else in your environment, connecto's root IDs
    can come back subtly wrong rather than obviously broken.

## Optional extras

Some annotation sources are lab-internal or need an extra package:

```bash
pip install "connecto[voxels]"    # compiled decoders for dense reads
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
                 name                          label                  species        backends                     annotations  public
0               aedes         Aedes (mosquito brain)            Aedes aegypti            cave                        flytable   False
1                banc    BANC (brain and nerve cord)  Drosophila melanogaster  neuprint, cave                  cave, flytable    True
2                fanc              FANC (female VNC)  Drosophila melanogaster            cave                            cave   False
3               fish2       fish2 (larval zebrafish)              Danio rerio        neuprint                        neuprint   False
4             flywire  FlyWire (FAFB) public release  Drosophila melanogaster  neuprint, cave                public, flytable    True
5  flywire-production      FlyWire (FAFB) production  Drosophila melanogaster            cave                public, flytable   False
6           hemibrain                      hemibrain  Drosophila melanogaster        neuprint                        neuprint    True
7             malecns                       male CNS  Drosophila melanogaster        neuprint                  neuprint, clio    True
8                manc                MANC (male VNC)  Drosophila melanogaster        neuprint                        neuprint    True
9             microns      MICrONS (minnie65) public             Mus musculus            cave  celltypes, mtypes, nucleus_svm    True
```

The first backend listed is the default. `public=False` does not mean secret — it means a
fresh token will not get you in, and `cn.get_spec(name).access` says what will.

That call is offline — it only reads the registry. To actually *query* anything you
need tokens, which is the [next page](credentials.md).
