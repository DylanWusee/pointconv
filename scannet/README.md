## ScanNet v2 Data

Please download original dataset from weibsite: <a href="http://www.scan-net.org/">http://www.scan-net.org/</a>

To prepare the Scannet dataset for training and evaluation, modity [line 82](https://github.com/DylanWusee/pointconv/blob/2a59507ef8798d52225885865ecc4b50face78c9/scannet/scannetv2_seg_dataset_rgb21c_pointid.py#L82) in `scannetv2_seg_dataset_rgb21c_pointid.py` to your ScanNet v2 dataset path.

Then,

```
python scannetv2_seg_dataset_rgb21c_pointid.py
```

This will generate three pickle files: `scannet_train_rgb21c_pointid.pickle`, `scannet_val_rgb21c_pointid.pickle`, and `scannet_test_rgb21c_pointid.pickle`. The first two are used in training and validation.
