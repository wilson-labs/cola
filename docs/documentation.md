# Update documentation

To rebuild the documentation, you need to install the requirement packages:
```
pip install -r docs/requirements.txt
```
And then run:
```
sphinx-build -b html docs docs/build/html
```
This can take a long time because it executes many of the notebooks in the documentation source;
if you'd prefer to build the docs without exeuting the notebooks, you can run:
```
sphinx-build -b html -D jupyter_execute_notebooks=off docs docs/build/html
```
You can then see the generated documentation in `docs/build/html/index.html`.


### Editing ipynb

To edit notebooks in the Colab interface,
open <http://colab.research.google.com> and `Upload` from your local repo.
Update it as needed, `Run all cells` then `Download ipynb` 
You may want to test that it executes properly, using `sphinx-build` as explained above.