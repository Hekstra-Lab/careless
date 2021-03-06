import pytest
from careless.merge.merge import *
from careless.utils.io import load_isomorphous_mtzs
from os.path import abspath,dirname,exists
import reciprocalspaceship as rs

from careless.utils.device import disable_gpu
status = disable_gpu()
assert status

dmin = 5. #Use less memory and go faster the test files only go out to 4 angstroms anyway

base_dir = dirname(abspath(__file__))
mtz_filenames = [
    base_dir + '/pyp_off.mtz',
    base_dir + '/pyp_2ms.mtz',
]

mtz_data = []
for f in mtz_filenames:
    m = rs.read_mtz(f)
    mtz_data.append(m[m.compute_dHKL().dHKL >= dmin])

reference_filename = base_dir + '/pyp_reference.mtz'

for f in mtz_filenames:
    assert exists(f)
assert exists(reference_filename)

reference_data = rs.read_mtz(reference_filename)
reference_data = reference_data[reference_data.compute_dHKL().dHKL >= dmin]

@pytest.mark.parametrize("merger_class", [MonoMerger, PolyMerger])
@pytest.mark.parametrize("anomalous", [False, True])
def test_Constructor(merger_class, anomalous):
    merger = merger_class.from_mtzs(*mtz_filenames, anomalous=anomalous)

@pytest.mark.parametrize("merger_class", [MonoMerger, PolyMerger])
@pytest.mark.parametrize("anomalous", [False, True])
@pytest.mark.parametrize("data", [reference_filename, reference_data])
def test_AppendReference(merger_class, data, anomalous):
    merger = merger_class.from_mtzs(*mtz_filenames, anomalous=anomalous)
    if merger_class == PolyMerger:
        merger.expand_harmonics()
    merger.append_reference_data(data)
    assert 'REF' in merger.data
    assert 'SIGREF' in merger.data


@pytest.mark.parametrize("merger_class", [MonoMerger, PolyMerger])
@pytest.mark.parametrize("metadata_keys", [None, ['dHKL', 'X', 'Y', 'BATCH']])
@pytest.mark.parametrize("anomalous", [False, True])
@pytest.mark.parametrize("likelihood", ["Normal", "Laplace", "StudentT"])
@pytest.mark.parametrize("prior", ["Wilson", "Normal", "Laplace", "StudentT"])
@pytest.mark.parametrize("mc_samples", [2])
@pytest.mark.parametrize("image_scaler", [True, False])
def test_Merger_on_reference_data(merger_class, anomalous, prior, likelihood, metadata_keys, mc_samples, image_scaler):
    merger = merger_class(mtz_data, anomalous=anomalous)
    if merger_class == PolyMerger:
        merger.expand_harmonics()

    assert merger.data[merger.sigma_intensity_key].dtype == 'Q'
    assert merger.data[merger.intensity_key].dtype == 'J'

    if prior != "Wilson":
        #The empirical priors need reference data
        merger.append_reference_data(reference_data)
        assert 'REF' in merger.data
        assert 'SIGREF' in merger.data

    merger.prep_indices()

    if prior == 'Normal':
        merger.add_normal_prior()
    elif prior == 'Laplace':
        merger.add_laplace_prior()
    elif prior == 'StudentT':
        merger.add_studentt_prior(4.)
    elif prior == "Wilson":
        merger.add_wilson_prior()
    else:
        raise ValueError(f"prior, {prior}, is not a valid selection")

    if likelihood == 'Normal':
        merger.add_normal_likelihood()
    elif likelihood == 'Laplace':
        merger.add_laplace_likelihood()
    elif likelihood == 'StudentT':
        merger.add_studentt_likelihood(4.)
    else:
        raise ValueError(f"prior, {prior}, is not a valid selection")


    merger.prior.log_prob(merger.prior.mean())
    merger.likelihood.sample()

    merger.add_scaling_model(layers=1, metadata_keys=metadata_keys)
    if image_scaler:
        merger.add_image_scaler(image_id_key='BATCH')

    for model in merger.scaling_model:
        model.sample()

    loss = merger.train_model(10, mc_samples=mc_samples)
    assert np.all(np.isfinite(loss))
