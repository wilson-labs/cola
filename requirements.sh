py -m pip install pytest plum-dispatch tqdm matplotlib pyqt5 seaborn pandas scikit-learn --no-cache-dir
py -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html --no-cache-dir
py -m pip install objax optax --no-cache-dir

# py -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu --no-cache-dir
py -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116 --no-cache-dir
# py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir
# py -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# py -m pip install --upgrade "jax[cpu]" --no-cache-dir
py -m pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --no-cache-dir
# py -m pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --no-cache-dir
# py -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# py -m pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --no-cache-dir
    # use the above command for previous versions on CUDA (like 11.6)
    # also it uses cudnn82 which succesfully worked for Mint
    # I also used this install for Gauss
