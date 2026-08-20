# Grand Challenge submission container

The algorithm container used for ISLES'26. It loads the three-scheme ensemble from
`/opt/ml/model` and applies the same post-processing as `../postprocess.py`.

```bash
./do_build.sh       # build the image
./do_test_run.sh    # run it locally against test/ inputs
./do_save.sh        # export image + model tarballs for upload
```
